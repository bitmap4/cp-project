# Predicting Human Reading Times: Causal vs. Masked LMs on Syntactic Ambiguity

Team: Ishaan Romil (2023114011), Abhyudit Singh (2023114009)

## Status

Mid-project deliverable covers:

- **Phase 1** — Feature extraction: GPT-2 surprisal + per-layer embeddings; BERT per-layer embeddings (layers 0–12).
- **Phase 2** — Regression modeling: Ridge / Lasso / MLP mapping features → mean reading time.
- **Phase 3** — Layer-wise analysis of BERT: per-layer R² / Spearman ρ curves vs. reading time.

Post-midterm: interpretability (probing, attention rollout, PCA/t-SNE) and targeted linguistic
analysis of garden-path / relative-clause regions.

## Setup

```bash
pip install -e .
```

## Pipeline

All entry points use [Hydra](https://hydra.cc/). Compute parameters (GPU/CPU count, batch size,
precision) are selectable via the `compute` group: `compute=local` or `compute=slurm`.

```bash
# 1. Download + preprocess Natural Stories Corpus
python scripts/download_data.py

# 2. Extract features (GPU-heavy; use SLURM)
python scripts/extract_features.py model=gpt2 compute=slurm
python scripts/extract_features.py model=bert compute=slurm

# 3. Train regression baselines
python scripts/train_regression.py model=gpt2 regression=ridge
python scripts/train_regression.py model=bert regression=ridge

# 4. BERT layer-wise analysis
python scripts/layerwise_analysis.py model=bert regression=ridge
```

### Running on SLURM (Ada / irel)

```bash
sbatch slurm/run.sbatch
```

The SLURM script mirrors `sinteractive -c 36 -g 4 -A irel`. CPU/GPU counts are sourced from
`conf/compute/slurm.yaml`, so overrides flow through Hydra (e.g. `compute.num_gpus=2`).

## Layout

```
conf/           Hydra configs (data / model / regression / compute)
src/            Library code
scripts/        Hydra entry points
slurm/          SLURM batch script
```
