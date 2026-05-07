"""Run spaCy over the processed word table and write syntactic region annotations."""
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data.syntactic_regions import annotate_corpus, write_regions_tsv


def _load_spacy(model_name: str):
    import spacy
    try:
        spacy.prefer_gpu()
    except Exception:
        pass
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    nlp = _load_spacy(cfg.get("spacy_model", "en_core_web_trf"))
    table = pd.read_parquet(Path(cfg.paths.processed_dir) / "word_table.parquet")
    regions = annotate_corpus(table, nlp)

    out_path = Path(cfg.h1.regions_file)
    write_regions_tsv(regions, out_path)

    counts = regions.groupby("region").size().to_dict()
    n_items = regions["item_id"].nunique()
    print(f"[annotate] stories={table['story'].nunique()} items={n_items} "
          f"rows={len(regions)} by_region={counts} -> {out_path}")


if __name__ == "__main__":
    main()
