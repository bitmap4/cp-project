#!/usr/bin/env bash
#SBATCH --job-name=cp-project
#SBATCH --account=irel
#SBATCH --partition=ihub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Mirrors: sinteractive -c 36 -g 4 -A irel
# Override node with `sbatch -w gnodeNNN slurm/run.sbatch`.

set -euo pipefail
mkdir -p logs

# Activate your environment here (edit as needed)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cp
# source .venv/bin/activate

echo "[slurm] job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Hydra compute group is overridden so num_gpus/num_cpus match what SLURM gave us.
COMPUTE_OVERRIDES="compute=slurm compute.num_gpus=${SLURM_GPUS_ON_NODE:-4} compute.num_cpus=${SLURM_CPUS_PER_TASK:-36}"

python scripts/download_data.py
python scripts/extract_features.py model=gpt2 $COMPUTE_OVERRIDES
python scripts/extract_features.py model=bert $COMPUTE_OVERRIDES
python scripts/train_regression.py  model=gpt2 regression=ridge
python scripts/train_regression.py  model=bert regression=ridge
python scripts/layerwise_analysis.py model=bert regression=ridge
python scripts/layerwise_analysis.py model=gpt2 regression=ridge
