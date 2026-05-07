"""Download + preprocess corpus data into a parquet token table.

Supports: natural_stories (auto-download), dundee (expects local files).
"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils import ensure_dir

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.data.name == "natural_stories":
        from src.data import build_word_table, download_corpus, load_rts, load_words
        rt_path, words_path = download_corpus(cfg.paths.raw_dir, cfg.data.rt_url, cfg.data.words_url)
        rts = load_rts(rt_path, cfg.data.rt_min, cfg.data.rt_max)
        words = load_words(words_path)
        table = build_word_table(rts, words, cfg.data.min_subjects)

    elif cfg.data.name == "dundee":
        from src.data.dundee import build_dundee_word_table, load_dundee_dir
        raw_dir = Path(cfg.paths.raw_dir) / "dundee"
        rts = load_dundee_dir(raw_dir, rt_min=cfg.data.rt_min, rt_max=cfg.data.rt_max)
        table = build_dundee_word_table(rts, min_subjects=cfg.data.min_subjects)

    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    out = ensure_dir(cfg.paths.processed_dir) / "word_table.parquet"
    table.to_parquet(out, index=False)
    print(f"[{cfg.data.name}] Wrote {len(table)} tokens across {table['story'].nunique()} texts -> {out}")


if __name__ == "__main__":
    main()
