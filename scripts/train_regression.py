"""Train a regression model from extracted features -> log mean RT.

By default uses the final layer of the selected model. Override with `layer=N`
or `layer=null` (concat all layers).
"""
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.models import build_splits, fit_and_evaluate
from src.utils import ensure_dir, set_seed


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    table = pd.read_parquet(Path(cfg.paths.processed_dir) / "word_table.parquet")
    feats = np.load(Path(cfg.paths.features_dir) / cfg.model.name / "features.npz")

    layer = cfg.get("layer", -1)  # default: final layer
    if layer is not None and layer < 0:
        layer = feats["embeddings"].shape[1] + layer

    bundle = build_splits(
        table, feats, layer,
        use_surprisal=cfg.regression.use_surprisal and cfg.model.compute_surprisal,
        use_baseline=cfg.regression.use_baseline_features,
        val_stories=list(cfg.data.val_stories),
        test_stories=list(cfg.data.test_stories),
        standardize=cfg.regression.standardize,
    )
    print(f"[train] X_train={bundle.X_train.shape} features={len(bundle.feature_names)}")

    result = fit_and_evaluate(bundle, cfg.regression)
    result["model"] = cfg.model.name
    result["regression"] = cfg.regression.name
    result["layer"] = layer

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / cfg.model.name / cfg.regression.name)
    with open(out_dir / f"layer_{layer}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
