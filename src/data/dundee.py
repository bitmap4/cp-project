"""Dundee Eye-Tracking Corpus loader.

The Dundee Corpus (Kennedy & Pynte, 2005) contains eye-tracking data from
10 English-speaking participants reading 20 newspaper editorials (~51k tokens).

We use gaze duration (GD) as the primary reading-time measure, analogous to
self-paced reading time in Natural Stories. The loader expects either:

  (a) A single TSV/CSV file with columns:
      word, text_id, word_pos, participant, GD
      (additional columns like FFD, TRT are loaded if present)

  (b) Separate per-participant files in data/raw/dundee/ following the
      naming convention sa01.dat ... sa10.dat (standard Dundee distribution).

The output is a word_table DataFrame identical in schema to Natural Stories:
    story, zone, word, n_subjects, mean_rt, log_mean_rt, word_len, log_word_len
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir

log = logging.getLogger(__name__)

RT_COL_PRIORITY = ["GD", "GAZE_DURATION", "gaze_duration", "Gaze",
                    "FFD", "FIRST_FIXATION_DURATION", "first_fixation_duration"]


def _find_rt_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in RT_COL_PRIORITY:
        if c in df.columns:
            return c
    raise ValueError(f"No recognized RT column in {list(df.columns)}. "
                     f"Expected one of {RT_COL_PRIORITY}")


def _find_col(df: pd.DataFrame, candidates: list[str], fallback: str | None = None) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if fallback:
        return fallback
    raise ValueError(f"None of {candidates} found in {list(df.columns)}")


def load_dundee_tsv(path: str | Path, rt_col: str | None = None,
                    rt_min: float = 50, rt_max: float = 3000) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dundee data not found: {path}")

    sep = "\t" if path.suffix in (".tsv", ".dat") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]

    word_col = _find_col(df, ["word", "WORD", "Word", "token", "TOKEN"])
    text_col = _find_col(df, ["text_id", "TEXT_ID", "text", "item", "story"])
    pos_col = _find_col(df, ["word_pos", "WORD_POS", "wnum", "zone", "word_num", "WORD_NUM"])
    subj_col = _find_col(df, ["participant", "PARTICIPANT", "subject", "subj", "SUBJ", "WorkerId"])
    rt_col_name = _find_rt_col(df, rt_col)

    out = df.rename(columns={
        word_col: "word", text_col: "story", pos_col: "zone",
        subj_col: "subject", rt_col_name: "rt",
    })
    out["word"] = out["word"].astype(str)
    out["story"] = out["story"].astype(int)
    out["zone"] = out["zone"].astype(int)
    out["rt"] = pd.to_numeric(out["rt"], errors="coerce")

    out = out.dropna(subset=["rt"])
    out = out[(out["rt"] >= rt_min) & (out["rt"] <= rt_max)]
    return out[["word", "story", "zone", "subject", "rt"]]


def load_dundee_dir(raw_dir: str | Path, rt_min: float = 50, rt_max: float = 3000) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    tsv_files = sorted(raw_dir.glob("*.tsv")) + sorted(raw_dir.glob("*.csv"))
    if len(tsv_files) == 1:
        return load_dundee_tsv(tsv_files[0], rt_min=rt_min, rt_max=rt_max)
    if len(tsv_files) > 1:
        dfs = [load_dundee_tsv(f, rt_min=rt_min, rt_max=rt_max) for f in tsv_files]
        return pd.concat(dfs, ignore_index=True)

    dat_files = sorted(raw_dir.glob("*.dat"))
    if dat_files:
        dfs = []
        for f in dat_files:
            try:
                dfs.append(load_dundee_tsv(f, rt_min=rt_min, rt_max=rt_max))
            except Exception as e:
                log.warning("Skipping %s: %s", f, e)
        if dfs:
            return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(
        f"No Dundee data files found in {raw_dir}. "
        f"Place a TSV/CSV file with columns (word, text_id, word_pos, participant, GD) "
        f"in {raw_dir}/, or see the README for data download instructions."
    )


def build_dundee_word_table(
    rts: pd.DataFrame,
    min_subjects: int = 3,
) -> pd.DataFrame:
    agg = (
        rts.groupby(["story", "zone"], as_index=False)
        .agg(
            word=("word", "first"),
            mean_rt=("rt", "mean"),
            n_subjects=("subject", "nunique"),
        )
    )
    agg = agg[agg["n_subjects"] >= min_subjects]
    agg = agg.sort_values(["story", "zone"]).reset_index(drop=True)
    agg["log_mean_rt"] = np.log(agg["mean_rt"].clip(lower=1))
    agg["word_len"] = agg["word"].str.len().astype(int)
    agg["log_word_len"] = np.log(agg["word_len"].clip(lower=1))
    return agg
