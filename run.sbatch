#!/bin/bash
#SBATCH -c 32
#SBATCH -G 4
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL           # Send email notifications for all events
#SBATCH -w gnode062               # Specify the compute node
#SBATCH -o ./dp_output_%j.out     # Use job ID in output filename for uniqueness

set -euo pipefail

# sbatch launches from wherever you submitted, but all paths in the pipeline
# are relative to the project root. Pin it explicitly so re-submits are
# reproducible regardless of where `sbatch` was invoked from.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')")}"

mkdir -p logs

# Activate the uv-managed venv inside the SLURM job (sbatch doesn't inherit
# your login shell's activation). If the venv ever falls out of sync with
# pyproject.toml, run `uv sync` from the project root before submitting.
source .venv/bin/activate

# The project isn't pip-installed (no [build-system] in pyproject.toml),
# so Python can't find the `src` package from scripts/ via the venv alone.
# Put the project root on PYTHONPATH so `from src.data import ...` resolves.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "[slurm] job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Hydra compute group is overridden so num_gpus/num_cpus match what SLURM gave us.
COMPUTE_OVERRIDES="compute=slurm compute.num_gpus=${SLURM_GPUS_ON_NODE:-4} compute.num_cpus=${SLURM_CPUS_PER_TASK:-36}"

python scripts/download_data.py

# spaCy-based syntactic region annotations for H1 (relative clauses, reduced
# relatives, NP/S complement ambiguities + POS-matched controls). Uses the
# transformer pipeline on GPU; falls back to CPU if spacy.prefer_gpu() fails.
python -m spacy download en_core_web_trf || true
python scripts/annotate_regions.py

# Feature extraction (embeddings + surprisal). BERT pseudo-log-likelihood
# surprisal is enabled in conf/model/bert.yaml and is the expensive step.
python scripts/extract_features.py model=gpt2 $COMPUTE_OVERRIDES
python scripts/extract_features.py model=bert $COMPUTE_OVERRIDES

# Overall RT-prediction regressions (final layer by default).
python scripts/train_regression.py  model=gpt2 regression=ridge
python scripts/train_regression.py  model=bert regression=ridge

# Layerwise curves (H2 infrastructure, informative for both models).
python scripts/layerwise_analysis.py model=bert regression=ridge
python scripts/layerwise_analysis.py model=gpt2 regression=ridge

# H1: paired GPT-2 vs BERT comparison sliced by ambiguity region.
python scripts/h1_region_compare.py regression=ridge
