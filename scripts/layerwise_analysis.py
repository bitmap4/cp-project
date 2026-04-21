"""Per-layer predictive-score curve: fit regression for each layer independently.

Produces results/{model}/layerwise_{regression}.json and a PNG curve.
"""
import json
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.models import build_splits, fit_and_evaluate
from src.utils import ensure_dir, set_seed


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    table = pd.read_parquet(Path(cfg.paths.processed_dir) / "word_table.parquet")
    feats = np.load(Path(cfg.paths.features_dir) / cfg.model.name / "features.npz")
    num_layers = feats["embeddings"].shape[1]
    print(f"[layerwise] model={cfg.model.name} layers={num_layers}")

    per_layer = []
    for layer in range(num_layers):
        bundle = build_splits(
            table, feats, layer,
            use_surprisal=cfg.regression.use_surprisal and cfg.model.compute_surprisal,
            use_baseline=cfg.regression.use_baseline_features,
            val_stories=list(cfg.data.val_stories),
            test_stories=list(cfg.data.test_stories),
            standardize=cfg.regression.standardize,
        )
        res = fit_and_evaluate(bundle, cfg.regression)
        per_layer.append({"layer": layer, "val_r2": res["val"]["r2"], "val_spearman": res["val"]["spearman"],
                          "test_r2": res["test"]["r2"], "test_spearman": res["test"]["spearman"]})
        print(f"  layer {layer:>2}: val_r2={res['val']['r2']:.4f} test_r2={res['test']['r2']:.4f}")

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / cfg.model.name)
    out_json = out_dir / f"layerwise_{cfg.regression.name}.json"
    with open(out_json, "w") as f:
        json.dump(per_layer, f, indent=2)

    layers = [r["layer"] for r in per_layer]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, [r["val_r2"] for r in per_layer], marker="o", label="val R²")
    ax.plot(layers, [r["test_r2"] for r in per_layer], marker="s", label="test R²")
    ax.plot(layers, [r["val_spearman"] for r in per_layer], marker="^", linestyle="--", label="val ρ")
    ax.set_xlabel("Layer"); ax.set_ylabel("Score")
    ax.set_title(f"{cfg.model.name} layer-wise prediction of mean RT")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"layerwise_{cfg.regression.name}.png", dpi=150)
    print(f"[layerwise] wrote {out_json}")


if __name__ == "__main__":
    main()
