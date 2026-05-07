#!/bin/bash
#SBATCH -c 32
#SBATCH -G 4
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH -w gnode058
#SBATCH -o ./dp_output_%j.out

set -euo pipefail

# =========================================================================
# Move everything to /scratch and pin all caches there
# =========================================================================
SCRATCH="/scratch/$USER/cp_project"
mkdir -p "$SCRATCH"
rsync -a --exclude='.venv' --exclude='.git' "$SLURM_SUBMIT_DIR/" "$SCRATCH/"
cd "$SCRATCH"

export UV_CACHE_DIR="/scratch/$USER/.cache/uv"
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/transformers"
export TORCH_HOME="/scratch/$USER/.cache/torch"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
export SPACY_DATA="/scratch/$USER/.cache/spacy"
export XDG_CACHE_HOME="/scratch/$USER/.cache"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$SPACY_DATA"

module load u22/cuda/12.4
module load u22/cudnn/9.1.0-cuda-12
module load u22/python/3.12.4

echo "[slurm] job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=$CUDA_VISIBLE_DEVICES"
echo "[slurm] workdir=$SCRATCH"
nvidia-smi || true

uv sync --no-build-isolation

COMPUTE_OVERRIDES="compute=slurm compute.num_gpus=${SLURM_GPUS_ON_NODE:-4} compute.num_cpus=${SLURM_CPUS_PER_TASK:-36}"

# =========================================================================
# Phase 1: Natural Stories pipeline (skips steps whose outputs exist)
# =========================================================================
echo "=== Phase 1: Natural Stories ==="

if [ ! -f data/processed/word_table.parquet ]; then
    uv run python scripts/download_data.py
else
    echo "[skip] word_table.parquet already exists"
fi

if ! uv run python -c "import en_core_web_trf" 2>/dev/null; then
    uv run python -m spacy download en_core_web_trf || true
fi
if ! uv run python -c "import en_core_web_sm" 2>/dev/null; then
    uv run python -m spacy download en_core_web_sm || true
fi

if [ ! -s data/annotations/syntactic_regions.tsv ] || [ "$(wc -l < data/annotations/syntactic_regions.tsv)" -le 3 ]; then
    uv run python scripts/annotate_regions.py
else
    echo "[skip] syntactic_regions.tsv already populated"
fi

if [ ! -f features/gpt2/features.npz ]; then
    uv run python scripts/extract_features.py model=gpt2 $COMPUTE_OVERRIDES
else
    echo "[skip] features/gpt2/features.npz already exists"
fi

if [ ! -f features/bert/features.npz ]; then
    uv run python scripts/extract_features.py model=bert $COMPUTE_OVERRIDES
else
    echo "[skip] features/bert/features.npz already exists"
fi

# uv run python scripts/train_regression.py  model=gpt2 regression=ridge
# uv run python scripts/train_regression.py  model=bert regression=ridge

# uv run python scripts/layerwise_analysis.py model=bert regression=ridge
# uv run python scripts/layerwise_analysis.py model=gpt2 regression=ridge

# uv run python scripts/h1_region_compare.py regression=ridge

# =========================================================================
# Phase 4: Interpretability (PCA/t-SNE + probing classifiers)
# =========================================================================
echo "=== Phase 4: Interpretability ==="

uv run python scripts/interpretability.py model=bert best_layer=5
uv run python scripts/interpretability.py model=gpt2 best_layer=4

# =========================================================================
# Phase 5: Dundee corpus cross-validation (if data is available)
# =========================================================================
DUNDEE_DIR="data/raw/dundee"
if [ -d "$DUNDEE_DIR" ] && [ "$(ls -A "$DUNDEE_DIR" 2>/dev/null)" ]; then
    echo "=== Phase 5: Dundee Corpus ==="

    uv run python scripts/download_data.py data=dundee
    uv run python scripts/extract_features.py model=gpt2 data=dundee $COMPUTE_OVERRIDES
    uv run python scripts/extract_features.py model=bert data=dundee $COMPUTE_OVERRIDES
    uv run python scripts/train_regression.py  model=gpt2 data=dundee regression=ridge
    uv run python scripts/train_regression.py  model=bert data=dundee regression=ridge
    uv run python scripts/layerwise_analysis.py model=bert data=dundee regression=ridge
    uv run python scripts/layerwise_analysis.py model=gpt2 data=dundee regression=ridge
    uv run python scripts/interpretability.py model=bert data=dundee best_layer=5
    uv run python scripts/interpretability.py model=gpt2 data=dundee best_layer=4
else
    echo "[skip] Dundee data not found in $DUNDEE_DIR — skipping cross-dataset validation."
    echo "       Place Dundee TSV/CSV in $DUNDEE_DIR/ and re-run to include it."
fi

# =========================================================================
# Copy results back to submit dir
# =========================================================================
echo "=== Copying results back ==="
rsync -a results/ "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true
rsync -a features/ "$SLURM_SUBMIT_DIR/features/" 2>/dev/null || true
rsync -a data/processed/ "$SLURM_SUBMIT_DIR/data/processed/" 2>/dev/null || true
rsync -a data/annotations/ "$SLURM_SUBMIT_DIR/data/annotations/" 2>/dev/null || true

echo "=== Pipeline complete ==="
