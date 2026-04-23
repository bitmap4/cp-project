# Predicting Human Reading Times: Causal vs. Masked LMs on Syntactic Ambiguity

Team: Ishaan Romil (2023114011), Abhyudit Singh (2023114009)

## Research questions

- **H1 (Causal vs. Masked).** GPT-2 correlates better with human RTs at the *onset*
  of syntactic ambiguity; BERT correlates better in the *disambiguation* region
  thanks to bidirectional context.
- **H2 (Layer-wise).** Middle BERT layers (roughly 4-8) predict human RTs better
  than early or late layers.

## Pipeline at a glance

```
download_data.py                       (Natural Stories Corpus -> word table)
  |
  +-> annotate_regions.py        (spaCy -> syntactic region TSV for H1)
  |
  v
extract_features.py   model=gpt2       (GPT-2 surprisal + per-layer embeddings)
extract_features.py   model=bert       (BERT PLL surprisal + per-layer embeddings)
  |
  +-> train_regression.py        (final-layer Ridge/Lasso/MLP -> log RT)
  |
  +-> layerwise_analysis.py      (H2: per-layer R^2 / rho curves)
  |
  +-> h1_region_compare.py       (H1: paired GPT-2 vs BERT by region)
```

## Setup

```bash
# uv-managed environment (recommended)
uv sync                    # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate  # or: use `uv run python ...` per command
```

All entry points use [Hydra](https://hydra.cc/). The `compute` group selects
CPU/GPU/batch-size defaults: `compute=local` (dev) or `compute=slurm` (cluster).

## Running

### Locally

```bash
python scripts/download_data.py
python -m spacy download en_core_web_trf    # once
python scripts/annotate_regions.py
python scripts/extract_features.py model=gpt2
python scripts/extract_features.py model=bert
python scripts/train_regression.py  model=gpt2 regression=ridge
python scripts/train_regression.py  model=bert regression=ridge
python scripts/layerwise_analysis.py model=bert regression=ridge
python scripts/layerwise_analysis.py model=gpt2 regression=ridge
python scripts/h1_region_compare.py  regression=ridge
```

### On SLURM

```bash
sbatch run.sbatch
```

`run.sbatch` activates `.venv`, sets `compute=slurm`, and runs the full pipeline.
Output lands in `./dp_output_<jobid>.out`.

### Expected runtime ($4\times$ RTX 2080 Ti, Natural Stories = 10 stories / ~10k tokens)

| Step | Time |
|---|---|
| `download_data.py` | ~10 s (one-off, cached thereafter) |
| `spacy download en_core_web_trf` | 1-2 min (one-off, ~450 MB) |
| `annotate_regions.py` (GPU, trf) | ~30 s |
| `extract_features.py model=gpt2` | ~1 min |
| `extract_features.py model=bert` (incl. PLL) | **5-10 min** (dominant step: one masked forward per subword) |
| `train_regression.py` $\times 2$ | <1 min |
| `layerwise_analysis.py` $\times 2$ | ~2-5 min |
| `h1_region_compare.py` | <1 min |
| **Total** | **~12-20 min** |

The pipeline is GPU-serial - each step uses one GPU. The 4-GPU reservation is
headroom for the PLL forward passes and a safety margin against OOM; there's
no multi-GPU parallelism in the current code. If the total ever matters, the
obvious win is parallelising `extract_features gpt2` and `bert` on two GPUs;
it's not engineered now because <20 min end-to-end isn't a bottleneck.

## Repository layout

```
conf/            Hydra configs
  config.yaml      top-level config, paths, H1 settings
  compute/         local.yaml, slurm.yaml
  data/            natural_stories.yaml
  model/           gpt2.yaml, bert.yaml
  regression/      ridge.yaml, lasso.yaml, mlp.yaml
src/             library code (imported by scripts)
  data/            Natural Stories loader + region annotations
  features/        GPT-2 / BERT feature extraction
  models/          regression (build_splits, fit_and_evaluate)
  utils.py         seeding, device selection, I/O
scripts/         Hydra entry points
data/            downloaded + processed corpora (gitignored); annotations/ is tracked
features/        per-model features.npz (created by extract_features.py)
results/         JSON / PNG outputs from train / layerwise / h1 scripts
run.sbatch       SLURM launcher for the full pipeline
```

## File-by-file description

### Configuration (`conf/`)

| File | Purpose |
|---|---|
| `conf/config.yaml` | Root config. Defines `paths.*`, the H1 section (regions file, permutation-test settings), random seed, and the default composition (`data: natural_stories`, `model: gpt2`, `regression: ridge`, `compute: local`). |
| `conf/compute/local.yaml` | Single-GPU dev defaults (`batch_size=8`, `num_cpus=4`). |
| `conf/compute/slurm.yaml` | Cluster defaults (`batch_size=32`, `num_cpus=36`, `mixed_precision=true`). Overridden at submit time from `$SLURM_*` vars. |
| `conf/data/natural_stories.yaml` | Upstream URLs for the Natural Stories RT/word files, RT filtering bounds, `min_subjects` threshold, and the `val_stories` / `test_stories` split. |
| `conf/model/gpt2.yaml` | `hf_id=gpt2`, 12 hidden layers, sliding-window parameters for long-context surprisal, `compute_surprisal: true`. |
| `conf/model/bert.yaml` | `hf_id=bert-base-uncased`, 12 layers, `compute_surprisal: true` (pseudo-log-likelihood), `pll_batch_size`. |
| `conf/regression/ridge.yaml` | Ridge $\alpha$ grid, CV folds, and flags for baseline features / surprisal / standardization. |
| `conf/regression/lasso.yaml` | Same schema, Lasso $\alpha$ grid. |
| `conf/regression/mlp.yaml` | Lightweight MLP hyperparameters (hidden sizes, dropout, LR, early stopping). |

### Source (`src/`)

| File | Purpose |
|---|---|
| `src/utils.py` | `set_seed`, `resolve_device` (auto/cuda/mps/cpu), `ensure_dir`, `save_npz`. |
| `src/data/natural_stories.py` | Downloads the RT TSV and tokenized story file; `load_rts` filters by RT bounds and correct-trial flag; `load_words` normalises columns; `build_word_table` aggregates per-subject RTs to (story, zone) mean RTs plus `log_mean_rt`, `word_len`, `log_word_len`; `split_by_story` holds out val/test stories. |
| `src/data/regions.py` | Ambiguity-region machinery for H1. `load_region_annotations` reads a regions TSV; `expand_regions_to_tokens` joins spans to per-token labels; `auto_regions_from_surprisal` is the surprisal-based proxy used when no annotations are supplied. |
| `src/data/syntactic_regions.py` | spaCy-based auto-annotator. `annotate_story` walks the parsed sentence and emits onset/disambig/control rows for (i) relative clauses (head noun + RC verb), (ii) reduced relatives (head noun + past participle), (iii) NP/S complement ambiguities (post-verb NP subject + embedded verb), plus POS-matched controls. `annotate_corpus` runs it over the whole word table; `write_regions_tsv` dumps a regions TSV. |
| `src/data/__init__.py` | Re-exports the data + regions API. |
| `src/features/extract.py` | `_align_subwords` maps each surface word to its subword span. `extract_causal` runs GPT-2 with a sliding window, produces per-layer embeddings (mean-pooled across subwords) and per-word surprisal (summed `-log p` across subwords). `extract_masked` runs BERT over striding windows, averaging hidden states in overlap; if `compute_surprisal=true`, `_bert_pll_surprisal` runs one masked forward pass per subword (batched) for Salazar-style pseudo-log-likelihood surprisal. |
| `src/features/__init__.py` | Re-exports `extract_causal`, `extract_masked`. |
| `src/models/regression.py` | `FeatureBundle` dataclass; `build_splits` joins the word table with a `features.npz`, picks a specific layer (or concats all), tacks on surprisal + baseline (length, log-freq, position), standardizes using the training-split scaler, and splits by story. `fit_and_evaluate` does $\alpha$-search for Ridge/Lasso on the val set, or trains the MLP with early stopping. `_score` returns $R^2$, Spearman $\rho$, RMSE. |
| `src/models/__init__.py` | Re-exports `build_splits`, `fit_and_evaluate`. |

### Scripts (`scripts/`)

| File | Purpose |
|---|---|
| `scripts/download_data.py` | Downloads RT + story files, builds the processed word table, writes `data/processed/word_table.parquet`. Idempotent. |
| `scripts/annotate_regions.py` | Runs spaCy over the word table and writes `data/annotations/syntactic_regions.tsv`. CPU-only, cheap. Downloads `en_core_web_sm` on first call. |
| `scripts/extract_features.py` | Loads the word table, runs the selected model's extractor, and writes `features/{model}/features.npz` containing `embeddings [N, num_layers, H]`, `surprisal [N]`, `story`, `zone`. Also dumps the resolved model config next to the features. |
| `scripts/train_regression.py` | Fits one regression on a chosen layer (default `-1`, override with `layer=<i>`), saves `results/{model}/{regression}/layer_<i>.json` with val / test $R^2$, Spearman $\rho$, RMSE, and $\alpha$. |
| `scripts/layerwise_analysis.py` | H2: fits one regression per layer, writes `results/{model}/layerwise_{regression}.json` and a PNG curve of val/test $R^2$ and val $\rho$ across layers. |
| `scripts/h1_region_compare.py` | H1: fits Ridge/Lasso on the training stories for *both* GPT-2 and BERT, aligns the two models on shared (story, zone) tokens, joins to region labels (hand-annotated or surprisal proxy), and reports per-region $R^2$ / Spearman $\rho$ / mean\|residual\| for each model plus a paired permutation test of $\lvert resid_{gpt2} \rvert - \lvert resid_{bert} \rvert$. Writes `results/h1/region_compare_{regression}.json` + PNG. Directional verdict fields: `gpt2_better_at_onset`, `bert_better_at_disambig`. |

### Data

| Path | Purpose |
|---|---|
| `data/raw/` | Raw downloaded corpus files (gitignored). |
| `data/processed/word_table.parquet` | Per-token mean RT + baseline features, keyed by (story, zone). |
| `data/annotations/syntactic_regions.tsv` | Auto-generated by `scripts/annotate_regions.py` from spaCy parses of Natural Stories. Covers relative clauses, reduced relatives, and NP/S complement ambiguities. The H1 default regions file. |
| `data/annotations/ambiguity_regions.tsv` | Hand-curated ambiguity regions. Same schema as `syntactic_regions.tsv`. Overrides the auto file if you point `h1.regions_file` at it. Columns: `story, item_id, region $\in$ \{onset, disambig, control\}, zone_start, zone_end, note`. Comment lines start with `#`. If both files are empty, H1 analysis falls back to a GPT-2-surprisal proxy. |

### Outputs

| Path | Produced by |
|---|---|
| `features/{gpt2,bert}/features.npz` | `extract_features.py` |
| `features/{gpt2,bert}/config.yaml` | `extract_features.py` (resolved model config for reproducibility) |
| `results/{model}/{regression}/layer_{i}.json` | `train_regression.py` |
| `results/{model}/layerwise_{regression}.{json,png}` | `layerwise_analysis.py` |
| `results/h1/region_compare_{regression}.{json,png}` | `h1_region_compare.py` |

## How each hypothesis is tested

**H1.** Natural Stories doesn't come with labelled garden-path items, so
`scripts/annotate_regions.py` runs spaCy over the corpus and marks the three
syntactic structures that drive GP-style ambiguity (relative clauses, reduced
relatives, NP/S complements). For each item we get an `onset` (the token where
parser uncertainty begins) and a `disambig` (the token that commits the
structure), plus a POS-matched `control`. Ridge is fit on training stories
for *both* GPT-2 and BERT, predictions are aligned on shared held-out tokens,
and mean absolute residuals are compared per region. A paired permutation
test on $\lvert resid_{gpt2} \rvert - \lvert resid_{bert} \rvert$ asks whether GPT-2 wins at `onset` and
whether BERT wins at `disambig`. Both models contribute embeddings + surprisal
(BERT's via masked pseudo-log-likelihood) so the comparison is symmetric.

**H2.** `layerwise_analysis.py` fits one regression per BERT layer (0-12) and
plots val / test $R^2$ and Spearman $\rho$. A middle-layer peak supports H2.

## Extending the analysis

- **Better region annotations.** Re-run `annotate_regions.py` after tweaking
  the detectors in `src/data/syntactic_regions.py`, or write hand annotations
  into `data/annotations/ambiguity_regions.tsv` and set
  `h1.regions_file=${paths.annotations_dir}/ambiguity_regions.tsv`.
- **Other regressors.** Pass `regression=lasso` or `regression=mlp`. For the
  H1 script, only Ridge/Lasso are supported (MLP residual interpretation is
  noisier; extend `_fit_predict_all` if needed).
- **Other layers.** `train_regression.py layer=6` picks layer 6; `layer=null`
  concatenates all layers.
- **Interpretability phase 4.** Not yet implemented. Planned: probing
  classifiers, attention rollout on the top H1 layer, and PCA/t-SNE on the
  best-performing BERT layer's activations.
