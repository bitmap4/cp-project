"""Automatic syntactic-region annotation for Natural Stories via spaCy.

Detects relative clauses, reduced relatives, and NP/S complement ambiguities,
marking onset / disambig zones plus POS-matched controls.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


NP_S_VERBS: set[str] = {
    "know", "knew", "believe", "believed", "think", "thought", "say", "said",
    "hear", "heard", "see", "saw", "realise", "realize", "realised", "realized",
    "understand", "understood", "expect", "expected", "suspect", "suspected",
    "assume", "assumed", "remember", "remembered", "forget", "forgot",
    "notice", "noticed", "claim", "claimed", "admit", "admitted",
    "decide", "decided", "find", "found", "feel", "felt", "show", "showed",
    "tell", "told", "prove", "proved", "suggest", "suggested",
}


def _build_char_to_zone(words: list[str], zones: list[int]) -> tuple[str, dict[int, int]]:
    parts: list[str] = []
    char_to_zone: dict[int, int] = {}
    offset = 0
    for w, z in zip(words, zones):
        for c in range(offset, offset + len(w)):
            char_to_zone[c] = z
        parts.append(w)
        offset += len(w) + 1
    return " ".join(parts), char_to_zone


def _token_zone(tok, char_to_zone: dict[int, int]) -> int | None:
    return char_to_zone.get(tok.idx)


def annotate_story(
    story_id: int,
    words: list[str],
    zones: list[int],
    nlp,
) -> list[dict]:
    text, char_to_zone = _build_char_to_zone(words, zones)
    doc = nlp(text)

    rows: list[dict] = []
    onset_zones: set[int] = set()
    disambig_zones: set[int] = set()

    for tok in doc:
        if tok.dep_ == "relcl" and tok.pos_ in {"VERB", "AUX"}:
            head_zone = _token_zone(tok.head, char_to_zone)
            verb_zone = _token_zone(tok, char_to_zone)
            if head_zone is None or verb_zone is None or head_zone == verb_zone:
                continue
            tag = f"s{story_id}_relcl_{tok.i}"
            rows.append({"story": story_id, "item_id": tag, "region": "onset",
                         "zone_start": head_zone, "zone_end": head_zone,
                         "note": f"RC head `{tok.head.text}`"})
            rows.append({"story": story_id, "item_id": tag, "region": "disambig",
                         "zone_start": verb_zone, "zone_end": verb_zone,
                         "note": f"RC verb `{tok.text}`"})
            onset_zones.add(head_zone)
            disambig_zones.add(verb_zone)

    for tok in doc:
        if tok.dep_ == "acl" and tok.tag_ == "VBN" and tok.pos_ == "VERB":
            head_zone = _token_zone(tok.head, char_to_zone)
            part_zone = _token_zone(tok, char_to_zone)
            if head_zone is None or part_zone is None or head_zone == part_zone:
                continue
            if head_zone in onset_zones or part_zone in disambig_zones:
                continue
            tag = f"s{story_id}_redrel_{tok.i}"
            rows.append({"story": story_id, "item_id": tag, "region": "onset",
                         "zone_start": head_zone, "zone_end": head_zone,
                         "note": f"RR head `{tok.head.text}`"})
            rows.append({"story": story_id, "item_id": tag, "region": "disambig",
                         "zone_start": part_zone, "zone_end": part_zone,
                         "note": f"RR participle `{tok.text}`"})
            onset_zones.add(head_zone)
            disambig_zones.add(part_zone)

    for tok in doc:
        if tok.lemma_.lower() not in NP_S_VERBS or tok.pos_ != "VERB":
            continue
        ccomp = next((c for c in tok.children if c.dep_ == "ccomp" and c.pos_ in {"VERB", "AUX"}), None)
        if ccomp is None:
            continue
        subj = next((c for c in ccomp.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
        if subj is None or subj.pos_ not in {"NOUN", "PROPN", "PRON"}:
            continue
        onset_zone = _token_zone(subj, char_to_zone)
        verb_zone = _token_zone(ccomp, char_to_zone)
        if onset_zone is None or verb_zone is None or onset_zone == verb_zone:
            continue
        if onset_zone in onset_zones or verb_zone in disambig_zones:
            continue
        tag = f"s{story_id}_nps_{tok.i}"
        rows.append({"story": story_id, "item_id": tag, "region": "onset",
                     "zone_start": onset_zone, "zone_end": onset_zone,
                     "note": f"NP/S onset `{subj.text}` under `{tok.text}`"})
        rows.append({"story": story_id, "item_id": tag, "region": "disambig",
                     "zone_start": verb_zone, "zone_end": verb_zone,
                     "note": f"NP/S disambig `{ccomp.text}`"})
        onset_zones.add(onset_zone)
        disambig_zones.add(verb_zone)

    already = onset_zones | disambig_zones
    n_items = len({r["item_id"] for r in rows})
    controls_onset = []
    controls_disambig = []
    for tok in doc:
        z = _token_zone(tok, char_to_zone)
        if z is None or z in already:
            continue
        if tok.pos_ in {"NOUN", "PROPN", "PRON"} and len(controls_onset) < n_items:
            controls_onset.append((z, tok))
            already.add(z)
        elif tok.pos_ in {"VERB", "AUX"} and len(controls_disambig) < n_items:
            controls_disambig.append((z, tok))
            already.add(z)
        if len(controls_onset) >= n_items and len(controls_disambig) >= n_items:
            break
    for i, (z, tok) in enumerate(controls_onset):
        rows.append({"story": story_id, "item_id": f"s{story_id}_ctrl_on_{i}",
                     "region": "control", "zone_start": z, "zone_end": z,
                     "note": f"control NOUN `{tok.text}`"})
    for i, (z, tok) in enumerate(controls_disambig):
        rows.append({"story": story_id, "item_id": f"s{story_id}_ctrl_di_{i}",
                     "region": "control", "zone_start": z, "zone_end": z,
                     "note": f"control VERB `{tok.text}`"})
    return rows


def annotate_corpus(table: pd.DataFrame, nlp) -> pd.DataFrame:
    all_rows: list[dict] = []
    for story_id, g in table.groupby("story"):
        g = g.sort_values("zone")
        rows = annotate_story(int(story_id), g["word"].tolist(), g["zone"].astype(int).tolist(), nlp)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows, columns=["story", "item_id", "region",
                                           "zone_start", "zone_end", "note"])


def write_regions_tsv(regions: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Auto-generated by scripts/annotate_regions.py\n")
        f.write("story\titem_id\tregion\tzone_start\tzone_end\tnote\n")
        for _, r in regions.iterrows():
            f.write(f"{r['story']}\t{r['item_id']}\t{r['region']}\t{r['zone_start']}\t{r['zone_end']}\t{r['note']}\n")
