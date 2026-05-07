"""H1: paired GPT-2 vs BERT comparison of RT-prediction by ambiguity region.

Fits regression on training stories for each model, compares held-out residuals
per region (onset / disambig / control) with a paired permutation test.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from src.data import (
    auto_regions_from_surprisal,
    expand_regions_to_tokens,
    load_region_annotations,
)
from src.utils import ensure_dir, set_seed


def _prepare_model_frame(
    table: pd.DataFrame,
    feats_path: Path,
    use_surprisal: bool,
    use_baseline: bool,
    layer: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    feats = np.load(feats_path)
    emb = feats["embeddings"]
    surp = feats["surprisal"]
    story = feats["story"]
    zone = feats["zone"]
    if layer < 0:
        layer = emb.shape[1] + layer

    feat_df = pd.DataFrame({"story": story, "zone": zone, "_row": np.arange(len(story))})
    merged = table.merge(feat_df, on=["story", "zone"], how="inner").reset_index(drop=True)
    rows = merged["_row"].to_numpy()

    E = emb[rows, layer, :]
    extras: list[np.ndarray] = []
    if use_surprisal and surp is not None and np.isfinite(surp).any():
        s = surp[rows].astype(np.float32)
        s = np.nan_to_num(s, nan=np.nanmean(s))
        extras.append(s[:, None])
    if use_baseline:
        extras.append(merged["log_word_len"].to_numpy()[:, None])
        counts = merged["word"].str.lower().value_counts()
        extras.append(np.log(merged["word"].str.lower().map(counts).to_numpy() + 1)[:, None])
        extras.append(merged["zone"].to_numpy()[:, None].astype(np.float32))

    X = np.concatenate([E] + extras, axis=1) if extras else E
    y = merged["log_mean_rt"].to_numpy(dtype=np.float32)
    return X, y, merged["story"].to_numpy(), merged


def _fit_predict_all(
    X: np.ndarray,
    y: np.ndarray,
    story: np.ndarray,
    val_stories: set[int],
    test_stories: set[int],
    cfg_reg: DictConfig,
) -> np.ndarray:
    train_mask = ~np.isin(story, list(val_stories | test_stories))
    val_mask = np.isin(story, list(val_stories))
    scaler = StandardScaler().fit(X[train_mask]) if cfg_reg.standardize else None
    Xs = scaler.transform(X) if scaler is not None else X

    if cfg_reg.kind not in ("ridge", "lasso"):
        raise ValueError(f"H1 comparison expects ridge/lasso, got {cfg_reg.kind}")
    best_est, best_score = None, -np.inf
    for a in cfg_reg.alphas:
        est = Ridge(alpha=a) if cfg_reg.kind == "ridge" else Lasso(alpha=a, max_iter=10_000)
        est.fit(Xs[train_mask], y[train_mask])
        score = est.score(Xs[val_mask], y[val_mask]) if val_mask.any() else est.score(Xs[train_mask], y[train_mask])
        if score > best_score:
            best_est, best_score = est, score
    return best_est.predict(Xs)


def _score(y: np.ndarray, yhat: np.ndarray) -> dict:
    if len(y) < 3:
        return {"n": int(len(y)), "r2": float("nan"), "spearman": float("nan"),
                "mean_abs_resid": float("nan")}
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho, _ = spearmanr(y, yhat)
    return {"n": int(len(y)), "r2": float(r2), "spearman": float(rho),
            "mean_abs_resid": float(np.mean(np.abs(y - yhat)))}


def _paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    d = a - b
    if len(d) < 3:
        return {"mean_diff": float(np.mean(d)) if len(d) else float("nan"),
                "p_value": float("nan"), "n": int(len(d))}
    obs = float(np.mean(d))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"mean_diff": obs, "p_value": p, "n": int(len(d))}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    table = pd.read_parquet(Path(cfg.paths.processed_dir) / "word_table.parquet")
    val_set = {int(s) for s in cfg.data.val_stories}
    test_set = {int(s) for s in cfg.data.test_stories}
    heldout = val_set | test_set

    feats_dir = Path(cfg.paths.features_dir)
    models = ["gpt2", "bert"]
    predictions = {}
    for m in models:
        feats_path = feats_dir / m / "features.npz"
        if not feats_path.exists():
            raise FileNotFoundError(f"Missing features for {m}: {feats_path}")
        feats = np.load(feats_path)
        X, y, story, merged = _prepare_model_frame(
            table, feats_path,
            use_surprisal=cfg.regression.use_surprisal and np.isfinite(feats["surprisal"]).any(),
            use_baseline=cfg.regression.use_baseline_features,
        )
        yhat = _fit_predict_all(X, y, story, val_set, test_set, cfg.regression)
        predictions[m] = {"y": y, "yhat": yhat, "story": story, "merged": merged}

    keys = predictions["gpt2"]["merged"][["story", "zone"]].merge(
        predictions["bert"]["merged"][["story", "zone"]], on=["story", "zone"], how="inner"
    )

    def _align(m: str) -> pd.DataFrame:
        df = predictions[m]["merged"][["story", "zone"]].copy()
        df["y"] = predictions[m]["y"]
        df[f"yhat_{m}"] = predictions[m]["yhat"]
        return df.merge(keys, on=["story", "zone"], how="inner").sort_values(["story", "zone"]).reset_index(drop=True)

    g = _align("gpt2")
    b = _align("bert")
    assert (g[["story", "zone"]].values == b[["story", "zone"]].values).all()
    aligned = g.rename(columns={"y": "y_gpt2"}).assign(yhat_bert=b["yhat_bert"], y_bert=b["y"])
    aligned["y"] = aligned["y_gpt2"]

    regions_path = Path(cfg.h1.regions_file)
    ann = load_region_annotations(regions_path)
    token_regions = expand_regions_to_tokens(ann, aligned[["story", "zone"]])
    region_source = "spacy_auto"
    if token_regions.empty and cfg.h1.auto_fallback:
        feats_g = np.load(feats_dir / "gpt2" / "features.npz")
        surp_full = pd.DataFrame({"story": feats_g["story"], "zone": feats_g["zone"], "surp": feats_g["surprisal"]})
        aligned_surp = aligned[["story", "zone"]].merge(surp_full, on=["story", "zone"], how="left")["surp"].to_numpy()
        token_regions = auto_regions_from_surprisal(
            aligned[["story", "zone"]].copy(), aligned_surp,
            onset_quantile=cfg.h1.auto_onset_quantile,
            disambig_offset=cfg.h1.auto_disambig_offset,
        )
        region_source = f"auto (gpt2 surprisal, q={cfg.h1.auto_onset_quantile})"

    df = aligned.merge(token_regions, on=["story", "zone"], how="left")
    df["region"] = df["region"].fillna("control")
    df["in_heldout"] = df["story"].isin(heldout)

    summary = {
        "region_source": region_source,
        "n_total": int(len(df)),
        "n_heldout": int(df["in_heldout"].sum()),
        "regions": {},
        "paired": {},
    }
    for region in ["onset", "disambig", "control"]:
        sub = df[(df["region"] == region) & df["in_heldout"]]
        s_gpt2 = _score(sub["y"].to_numpy(), sub["yhat_gpt2"].to_numpy())
        s_bert = _score(sub["y"].to_numpy(), sub["yhat_bert"].to_numpy())
        a = np.abs(sub["y"].to_numpy() - sub["yhat_gpt2"].to_numpy())
        b_arr = np.abs(sub["y"].to_numpy() - sub["yhat_bert"].to_numpy())
        perm = _paired_permutation(a, b_arr, int(cfg.h1.n_permutations), rng)
        summary["regions"][region] = {"gpt2": s_gpt2, "bert": s_bert}
        summary["paired"][region] = perm

    onset_diff = summary["paired"]["onset"]["mean_diff"]
    disambig_diff = summary["paired"]["disambig"]["mean_diff"]
    summary["h1_verdict"] = {
        "gpt2_better_at_onset": bool(onset_diff < 0),
        "bert_better_at_disambig": bool(disambig_diff > 0),
        "notes": "mean_diff = mean(|resid_gpt2| - |resid_bert|); negative favours GPT-2.",
    }

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / "h1")
    out_json = out_dir / f"region_compare_{cfg.regression.name}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    regions_plot = ["onset", "disambig", "control"]
    gpt_vals = [summary["regions"][r]["gpt2"]["mean_abs_resid"] for r in regions_plot]
    bert_vals = [summary["regions"][r]["bert"]["mean_abs_resid"] for r in regions_plot]
    x = np.arange(len(regions_plot))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, gpt_vals, width=0.4, label="GPT-2")
    ax.bar(x + 0.2, bert_vals, width=0.4, label="BERT")
    ax.set_xticks(x); ax.set_xticklabels(regions_plot)
    ax.set_ylabel("mean |residual| (log ms)")
    ax.set_title(f"H1: held-out residuals by region ({region_source})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"region_compare_{cfg.regression.name}.png", dpi=150)

    print(json.dumps(summary, indent=2))
    print(f"[h1] wrote {out_json}")


if __name__ == "__main__":
    main()
