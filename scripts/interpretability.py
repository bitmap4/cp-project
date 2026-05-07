"""Phase 4: Interpretability and activation analysis.

1. PCA / t-SNE on best-layer embeddings, colored by RT quartile, surprisal,
   and ambiguity region.
2. Per-layer probing classifiers for POS tag and dependency depth, testing
   which layers linearly encode linguistically meaningful features.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data import load_region_annotations, expand_regions_to_tokens
from src.utils import ensure_dir, set_seed


def _get_linguistic_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Run spaCy (small model, fast) to get POS tags and dependency depth."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    nlp.max_length = 2_000_000
    pos_tags = []
    dep_depths = []

    for story_id, g in table.groupby("story"):
        g = g.sort_values("zone")
        words = g["word"].tolist()
        text = " ".join(words)
        doc = nlp(text)

        tok_idx = 0
        char_offset = 0
        word_to_tok: dict[int, int] = {}
        for wi, w in enumerate(words):
            word_to_tok[wi] = tok_idx
            char_offset += len(w) + 1
            while tok_idx < len(doc) - 1 and doc[tok_idx + 1].idx < char_offset:
                tok_idx += 1
            tok_idx = min(tok_idx + 1, len(doc) - 1)

        for wi in range(len(words)):
            ti = min(word_to_tok.get(wi, 0), len(doc) - 1)
            tok = doc[ti]
            pos_tags.append(tok.pos_)
            depth = 0
            head = tok
            while head.head != head:
                depth += 1
                head = head.head
            dep_depths.append(depth)

    table = table.copy()
    table["pos"] = pos_tags
    table["dep_depth"] = dep_depths
    return table


def _visualize_embeddings(
    embeddings: np.ndarray,
    labels_dict: dict[str, np.ndarray],
    model_name: str,
    layer: int,
    out_dir: Path,
) -> None:
    scaler = StandardScaler()
    E = scaler.fit_transform(embeddings)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(E)

    n = min(5000, len(E))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(E), n, replace=False) if len(E) > n else np.arange(len(E))

    tsne = TSNE(n_components=2, perplexity=min(30, n // 4), random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(E[idx])

    for label_name, labels in labels_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        is_numeric = np.issubdtype(np.array(labels).dtype, np.floating) or np.issubdtype(np.array(labels).dtype, np.integer)

        if is_numeric:
            sc0 = axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap="viridis", s=4, alpha=0.5)
            plt.colorbar(sc0, ax=axes[0])
            sc1 = axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=np.array(labels)[idx], cmap="viridis", s=4, alpha=0.5)
            plt.colorbar(sc1, ax=axes[1])
        else:
            unique_labels = sorted(set(labels))
            colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 3)))
            for i, lab in enumerate(unique_labels):
                mask_all = np.array(labels) == lab
                axes[0].scatter(pca_2d[mask_all, 0], pca_2d[mask_all, 1],
                                c=[colors[i]], s=8, alpha=0.6, label=lab)
                mask_sub = np.array(labels)[idx] == lab
                axes[1].scatter(tsne_2d[mask_sub, 0], tsne_2d[mask_sub, 1],
                                c=[colors[i]], s=8, alpha=0.6, label=lab)
            axes[0].legend(fontsize=7, markerscale=2)
            axes[1].legend(fontsize=7, markerscale=2)

        axes[0].set_title(f"PCA — {model_name} L{layer} — {label_name}")
        axes[1].set_title(f"t-SNE — {model_name} L{layer} — {label_name}")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / f"{label_name}_L{layer}.png", dpi=150)
        plt.close(fig)


def _run_probing(
    embeddings_all_layers: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    is_classification: bool,
    pca_dim: int = 50,
) -> list[dict]:
    num_layers = embeddings_all_layers.shape[1]
    results = []
    for layer in range(num_layers):
        E = embeddings_all_layers[:, layer, :]
        scaler = StandardScaler()
        E_s = scaler.fit_transform(E)
        E_s = PCA(n_components=min(pca_dim, E_s.shape[1]), random_state=42).fit_transform(E_s)

        if is_classification:
            from sklearn.linear_model import SGDClassifier
            clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
            scores = cross_val_score(clf, E_s, labels, cv=3, scoring="accuracy")
            results.append({"layer": layer, "mean_accuracy": float(scores.mean()),
                            "std": float(scores.std())})
        else:
            reg = Ridge(alpha=100.0)
            scores = cross_val_score(reg, E_s, labels, cv=3, scoring="r2")
            results.append({"layer": layer, "mean_r2": float(scores.mean()),
                            "std": float(scores.std())})
        print(f"  [probe] {task_name} layer {layer:>2}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    table = pd.read_parquet(Path(cfg.paths.processed_dir) / "word_table.parquet")
    feats = np.load(Path(cfg.paths.features_dir) / cfg.model.name / "features.npz")
    embeddings = feats["embeddings"]
    surprisal = feats["surprisal"]
    num_layers = embeddings.shape[1]

    feat_df = pd.DataFrame({"story": feats["story"], "zone": feats["zone"],
                            "_row": np.arange(len(feats["story"]))})
    merged = table.merge(feat_df, on=["story", "zone"], how="inner").reset_index(drop=True)
    rows = merged["_row"].to_numpy()
    emb_aligned = embeddings[rows]
    surp_aligned = surprisal[rows]

    best_layer = cfg.get("best_layer", None)
    if best_layer is None:
        best_layer = 5 if cfg.model.name == "bert" else 4

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / cfg.model.name / "interpretability")

    # --- Linguistic labels via spaCy ---
    print(f"[interp] Getting linguistic labels via spaCy...")
    table_labeled = _get_linguistic_labels(merged)

    # --- 1. PCA / t-SNE visualization ---
    print(f"[interp] Running PCA/t-SNE on {cfg.model.name} layer {best_layer}...")
    E_best = emb_aligned[:, best_layer, :]

    rt_quartile = pd.qcut(merged["log_mean_rt"], q=4, labels=["Q1 (fast)", "Q2", "Q3", "Q4 (slow)"])

    has_surprisal = np.isfinite(surp_aligned).any()
    if has_surprisal:
        surp_clean = np.nan_to_num(surp_aligned, nan=np.nanmean(surp_aligned))
        surp_quartile = pd.qcut(surp_clean, q=4, labels=["low", "med-low", "med-high", "high"])

    regions_path = Path(cfg.h1.regions_file)
    ann = load_region_annotations(regions_path)
    token_regions = expand_regions_to_tokens(ann, merged[["story", "zone"]])
    token_regions_dedup = token_regions.drop_duplicates(subset=["story", "zone"], keep="first")
    region_labels = merged[["story", "zone"]].merge(
        token_regions_dedup, on=["story", "zone"], how="left"
    )["region"].fillna("control").to_numpy()

    labels_dict = {
        "rt_quartile": rt_quartile.to_numpy().astype(str),
        "ambiguity_region": region_labels,
        "pos_tag": table_labeled["pos"].to_numpy(),
    }
    if has_surprisal:
        labels_dict["surprisal_quartile"] = surp_quartile.to_numpy().astype(str)
    _visualize_embeddings(E_best, labels_dict, cfg.model.name, best_layer, out_dir)
    print(f"[interp] Saved PCA/t-SNE plots to {out_dir}")

    # --- 2. Probing classifiers ---
    print(f"[interp] Running probing classifiers across {num_layers} layers...")

    pos_labels = table_labeled["pos"].to_numpy()
    le_pos = LabelEncoder()
    pos_encoded = le_pos.fit_transform(pos_labels)
    pos_results = _run_probing(emb_aligned, pos_encoded, "POS", is_classification=True)

    dep_labels = table_labeled["dep_depth"].to_numpy().astype(float)
    dep_results = _run_probing(emb_aligned, dep_labels, "dep_depth", is_classification=False)

    rt_binary = (merged["log_mean_rt"] > merged["log_mean_rt"].median()).astype(int).to_numpy()
    rt_results = _run_probing(emb_aligned, rt_binary, "RT_binary", is_classification=True)

    all_results = {
        "model": cfg.model.name,
        "best_layer": best_layer,
        "pos_probing": pos_results,
        "dep_depth_probing": dep_results,
        "rt_binary_probing": rt_results,
        "pos_classes": le_pos.classes_.tolist(),
    }
    with open(out_dir / "probing_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Plot probing curves ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    layers = list(range(num_layers))

    axes[0].errorbar(layers, [r["mean_accuracy"] for r in pos_results],
                     yerr=[r["std"] for r in pos_results], marker="o", capsize=3)
    axes[0].set_title(f"{cfg.model.name}: POS probing accuracy")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Accuracy"); axes[0].grid(alpha=0.3)

    axes[1].errorbar(layers, [r["mean_r2"] for r in dep_results],
                     yerr=[r["std"] for r in dep_results], marker="s", capsize=3, color="orange")
    axes[1].set_title(f"{cfg.model.name}: Dep-depth probing R²")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("R²"); axes[1].grid(alpha=0.3)

    axes[2].errorbar(layers, [r["mean_accuracy"] for r in rt_results],
                     yerr=[r["std"] for r in rt_results], marker="^", capsize=3, color="green")
    axes[2].set_title(f"{cfg.model.name}: RT binary probing accuracy")
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Accuracy"); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "probing_curves.png", dpi=150)
    plt.close(fig)

    print(f"[interp] wrote {out_dir}/probing_results.json + probing_curves.png")


if __name__ == "__main__":
    main()
