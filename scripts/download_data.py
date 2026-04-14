"""Download + preprocess the Natural Stories Corpus into a parquet token table."""
import hydra
from omegaconf import DictConfig

from src.data import build_word_table, download_corpus, load_rts, load_words
from src.utils import ensure_dir


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    rt_path, words_path = download_corpus(cfg.paths.raw_dir, cfg.data.rt_url, cfg.data.words_url)
    rts = load_rts(rt_path, cfg.data.rt_min, cfg.data.rt_max)
    words = load_words(words_path)
    table = build_word_table(rts, words, cfg.data.min_subjects)

    out = ensure_dir(cfg.paths.processed_dir) / "word_table.parquet"
    table.to_parquet(out, index=False)
    print(f"Wrote {len(table)} tokens across {table['story'].nunique()} stories -> {out}")


if __name__ == "__main__":
    main()
