#!/bin/bash
#SBATCH --job-name=flexworld
#SBATCH --gres=gpu:1
# Full A100 80GB (not MIG 20GB slices); see `snodes` → gpu80 on adroit-h11g1
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-flexworld-%j.out

set -euo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

INPUT_DIR="${PROJECT_ROOT}/outputs/destylized"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/flexworld"
TRAJ="${1:-backward}"

module purge
module load anaconda3/2024.2

eval "$(conda shell.bash hook)"

echo "[INFO] Starting FlexWorld pipeline at $(date)"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Trajectory:   ${TRAJ}"

conda activate flexworld
bash "${PROJECT_ROOT}/run_flexworld.sh" "${INPUT_DIR}" "${OUTPUT_DIR}" "${TRAJ}"
conda deactivate

echo "[INFO] FlexWorld pipeline complete at $(date)"
