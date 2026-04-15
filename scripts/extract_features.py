"""Run feature extraction for a chosen model, save per-story .npz + manifest."""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.features import extract_causal, extract_masked
from src.utils import ensure_dir, resolve_device, save_npz, set_seed


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.compute.device)
    print(f"[extract] model={cfg.model.name} device={device}")

    table = pd.read_parquet(Path(cfg.paths.processed_dir) / "word_table.parquet")
    words_by_story = {s: g["word"].tolist() for s, g in table.groupby("story")}

    extract_layers = cfg.model.extract_layers
    if OmegaConf.is_list(extract_layers):
        extract_layers = list(extract_layers)

    if cfg.model.family == "causal":
        per_story = extract_causal(
            words_by_story, cfg.model.hf_id, device,
            cfg.model.context_window, cfg.model.stride,
            extract_layers, cfg.compute.mixed_precision,
        )
    elif cfg.model.family == "masked":
        per_story = extract_masked(
            words_by_story, cfg.model.hf_id, device,
            cfg.model.context_window, cfg.model.stride,
            extract_layers, cfg.compute.mixed_precision,
        )
    else:
        raise ValueError(f"Unknown model family: {cfg.model.family}")

    out_dir = ensure_dir(Path(cfg.paths.features_dir) / cfg.model.name)
    all_emb, all_surp, story_col, zone_col = [], [], [], []
    for story, g in table.groupby("story"):
        data = per_story[story]
        all_emb.append(data["embeddings"])
        all_surp.append(data["surprisal"] if data["surprisal"] is not None else np.full(len(g), np.nan, dtype=np.float32))
        story_col.append(np.full(len(g), story, dtype=np.int32))
        zone_col.append(g["zone"].to_numpy(dtype=np.int32))

    save_npz(
        out_dir / "features.npz",
        embeddings=np.concatenate(all_emb, axis=0),   # [N, num_layers, H]
        surprisal=np.concatenate(all_surp, axis=0),   # [N]
        story=np.concatenate(story_col),
        zone=np.concatenate(zone_col),
    )
    with open(out_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg.model))
    print(f"[extract] wrote {out_dir/'features.npz'}")


if __name__ == "__main__":
    main()
