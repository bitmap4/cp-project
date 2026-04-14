"""Natural Stories Corpus loader.

Upstream layout (languageMIT/naturalstories):
    naturalstories_RTS/processed_RTs.tsv  columns: WorkerId, WorkTimeInSeconds, correct, item, zone, RT
    naturalstories_RTS/all_stories.tok    columns: word, item, zone

`item` is the story id (1..10), `zone` is the word position within the story.
We aggregate per-subject RTs to a per-token mean RT, then attach the surface
word form, story id, and position.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.utils import ensure_dir

log = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    log.info("Downloading %s -> %s", url, dest)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def download_corpus(raw_dir: str | Path, rt_url: str, words_url: str) -> tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    rt_path = _download(rt_url, raw_dir / "processed_RTs.tsv")
    words_path = _download(words_url, raw_dir / "all_stories.tok")
    return rt_path, words_path


def load_rts(rt_path: str | Path, rt_min: float = 100, rt_max: float = 3000) -> pd.DataFrame:
    """Load per-subject reading times, filter extremes."""
    df = pd.read_csv(rt_path, sep="\t")
    # Normalise column names (corpus has shifted schemas across mirrors)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {"workerid": "subject", "subj": "subject", "item": "story", "storyid": "story"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df[(df["rt"] >= rt_min) & (df["rt"] <= rt_max)].copy()
    if "correct" in df.columns:
        # Only keep comprehension-question-correct trials when that column exists.
        df = df[df["correct"] == 1].copy()
    return df[["subject", "story", "zone", "rt"]]


def load_words(words_path: str | Path) -> pd.DataFrame:
    """Load surface tokens keyed by (story, zone)."""
    df = pd.read_csv(words_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {"item": "story", "storyid": "story"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["word"] = df["word"].astype(str)
    return df[["word", "story", "zone"]]


def build_word_table(
    rts: pd.DataFrame,
    words: pd.DataFrame,
    min_subjects: int = 5,
) -> pd.DataFrame:
    """Aggregate RTs per (story, zone) and attach word form + baseline features.

    Returns a DataFrame with one row per token, ordered by (story, zone), containing:
        story, zone, word, n_subjects, mean_rt, log_mean_rt, word_len, log_word_len
    """
    agg = (
        rts.groupby(["story", "zone"], as_index=False)
        .agg(mean_rt=("rt", "mean"), n_subjects=("subject", "nunique"))
    )
    agg = agg[agg["n_subjects"] >= min_subjects]

    out = words.merge(agg, on=["story", "zone"], how="inner").sort_values(["story", "zone"])
    out["log_mean_rt"] = np.log(out["mean_rt"])
    out["word_len"] = out["word"].str.len().astype(int)
    out["log_word_len"] = np.log(out["word_len"].clip(lower=1))
    return out.reset_index(drop=True)


def split_by_story(
    table: pd.DataFrame,
    val_stories: Iterable[int],
    test_stories: Iterable[int],
) -> dict[str, pd.DataFrame]:
    val_set = set(int(s) for s in val_stories)
    test_set = set(int(s) for s in test_stories)
    train = table[~table["story"].isin(val_set | test_set)].reset_index(drop=True)
    val = table[table["story"].isin(val_set)].reset_index(drop=True)
    test = table[table["story"].isin(test_set)].reset_index(drop=True)
    return {"train": train, "val": val, "test": test}
