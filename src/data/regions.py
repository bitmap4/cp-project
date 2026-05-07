"""Ambiguity-region annotations and surprisal-proxy fallback for H1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REGION_COLS = ["story", "item_id", "region", "zone_start", "zone_end"]
VALID_REGIONS = {"onset", "disambig", "control"}


def load_region_annotations(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REGION_COLS)
    rows: list[dict] = []
    with path.open() as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=REGION_COLS)
    df = pd.DataFrame(rows)
    for col in ("story", "zone_start", "zone_end"):
        df[col] = df[col].astype(int)
    df = df[df["region"].isin(VALID_REGIONS)].copy()
    return df[REGION_COLS]


def expand_regions_to_tokens(regions: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame(columns=["story", "zone", "region", "item_id"])
    out = []
    for _, r in regions.iterrows():
        mask = (
            (table["story"] == r["story"])
            & (table["zone"] >= r["zone_start"])
            & (table["zone"] <= r["zone_end"])
        )
        for _, t in table[mask].iterrows():
            out.append({"story": int(t["story"]), "zone": int(t["zone"]),
                        "region": r["region"], "item_id": r["item_id"]})
    return pd.DataFrame(out)


def auto_regions_from_surprisal(
    table: pd.DataFrame,
    surprisal: np.ndarray,
    onset_quantile: float = 0.9,
    disambig_offset: int = 1,
) -> pd.DataFrame:
    assert len(surprisal) == len(table), "surprisal length must match token table"
    tbl = table.reset_index(drop=True).copy()
    tbl["_surp"] = surprisal
    labels = np.full(len(tbl), "control", dtype=object)
    item_ids = np.full(len(tbl), "", dtype=object)
    next_id = 0
    for story, g in tbl.groupby("story"):
        idx = g.index.to_numpy()
        s = g["_surp"].to_numpy()
        valid = np.isfinite(s)
        if valid.sum() < 5:
            continue
        q = np.quantile(s[valid], onset_quantile)
        med = np.median(s[valid])
        onset_mask = valid & (s >= q)
        for local_i in np.where(onset_mask)[0]:
            global_i = idx[local_i]
            labels[global_i] = "onset"
            tag = f"auto_{next_id:04d}"
            item_ids[global_i] = tag
            tgt_local = local_i + disambig_offset
            if tgt_local < len(g) and np.isfinite(s[tgt_local]) and s[tgt_local] < med:
                global_j = idx[tgt_local]
                if labels[global_j] == "control":
                    labels[global_j] = "disambig"
                    item_ids[global_j] = tag
            next_id += 1
    out = tbl[["story", "zone"]].copy()
    out["region"] = labels
    out["item_id"] = item_ids
    return out
