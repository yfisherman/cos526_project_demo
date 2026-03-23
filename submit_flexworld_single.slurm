#!/bin/bash
#SBATCH --job-name=flexworld-single
#SBATCH --gres=gpu:1
# Full A100 80GB on adroit-h11g1 (see `snodes`: feature gpu80).
# Do NOT use bare `a100` alone — that can land on adroit-h11g2 MIG (3g.20gb ≈ 20GB) and OOM.
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-flexworld-single-%j.out

set -euo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"
FLEXWORLD_DIR="${PROJECT_ROOT}/FlexWorld"
CONDA_ENV="/scratch/network/${USER}/conda_envs/flexworld"
INPUT_IMAGE="${PROJECT_ROOT}/outputs/destylized/Gemini_Generated_Image_vi7dsfvi7dsfvi7d.png"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/flexworld/Gemini_Generated_Image_vi7dsfvi7dsfvi7d"
TRAJ="backward"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="/scratch/network/${USER}/tmp"
mkdir -p "${TMPDIR}"

module purge
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "[INFO] Starting FlexWorld at $(date)"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "[INFO] Input:      ${INPUT_IMAGE}"
echo "[INFO] Output:     ${OUTPUT_DIR}"
echo "[INFO] Trajectory: ${TRAJ}"

mkdir -p "${OUTPUT_DIR}"
cd "${FLEXWORLD_DIR}"

python video_generate.py \
  --input_image_path "${INPUT_IMAGE}" \
  --output_dir "${OUTPUT_DIR}" \
  --traj "${TRAJ}" \
  --name "Gemini_Generated_Image_vi7dsfvi7dsfvi7d"

echo "[INFO] FlexWorld complete at $(date)"
echo "[INFO] Output at: ${OUTPUT_DIR}"
