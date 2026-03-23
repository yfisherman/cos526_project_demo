#!/bin/bash
#SBATCH --job-name=cos526_phase1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

INPUT_DIR="${PROJECT_ROOT}/inputs/paintings"
DESTYLIZED_DIR="${PROJECT_ROOT}/outputs/destylized"
DEPTH_DIR="${PROJECT_ROOT}/outputs/depth_maps"
PROCESSED_DIR="${PROJECT_ROOT}/outputs/processed_inputs"
VIEW_OUT_DIR="${PROJECT_ROOT}/outputs/viewcrafter"

module purge
module load anaconda3/2024.2

eval "$(conda shell.bash hook)"

echo "[INFO] Starting Phase 1 pipeline at $(date)"
echo "[INFO] Project root: ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

conda activate cos526_destylize
python "${PROJECT_ROOT}/destylize.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${DESTYLIZED_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --processed_dir "${PROCESSED_DIR}" \
  --seed 42 \
  --strength 0.40 \
  --resize_mode pad \
  --num_inference_steps 40 \
  --controlnet_conditioning_scale 0.9 \
  --guidance_scale 5.0
conda deactivate

conda activate viewcrafter
bash "${PROJECT_ROOT}/run_viewcrafter.sh" "${DESTYLIZED_DIR}" "${VIEW_OUT_DIR}" "test/trajs/loop2.txt"
conda deactivate

echo "[INFO] Pipeline complete at $(date)"
