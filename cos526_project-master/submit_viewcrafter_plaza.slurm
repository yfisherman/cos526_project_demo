#!/bin/bash
#SBATCH --job-name=viewcrafter-plaza
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-viewcrafter-plaza-%j.out

set -eo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"

module purge
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"

conda activate viewcrafter

echo "[INFO] Starting ViewCrafter plaza run at $(date)"

bash "${PROJECT_ROOT}/run_viewcrafter.sh" \
  "${PROJECT_ROOT}/outputs/destylized_plaza_only" \
  "${PROJECT_ROOT}/outputs/viewcrafter_plaza" \
  "test/trajs/loop2.txt"

echo "[INFO] ViewCrafter plaza run complete at $(date)"
