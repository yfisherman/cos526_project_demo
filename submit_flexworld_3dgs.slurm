#!/bin/bash
#SBATCH --job-name=flexworld-3dgs
#SBATCH --gres=gpu:1
# Adroit: a100 alone can land on adroit-h11g2 (MIG 20GB slices → CogVideo OOM).
# gpu80 selects adroit-h11g1: 4× A100 80GB PCIe.
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-flexworld-3dgs-%j.out

set -euo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"
FLEXWORLD_DIR="${PROJECT_ROOT}/FlexWorld"
REPO_ROOT="${PROJECT_ROOT}"
# If cos526_project is only scripts, also support worktree copy:
if [ ! -f "${REPO_ROOT}/install_flexworld_addons.sh" ] && [ -f "${SLURM_SUBMIT_DIR}/install_flexworld_addons.sh" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

INPUT_IMAGE="${PROJECT_ROOT}/outputs/destylized/Gemini_Generated_Image_vi7dsfvi7dsfvi7d.png"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/flexworld_3dgs"
CONFIG_NAME="gemini_painting"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"

export TORCH_HOME="/scratch/network/${USER}/torch_hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

module purge
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"

echo "[INFO] FlexWorld main_3dgs (Gaussian splat) at $(date)"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ ! -d "${FLEXWORLD_DIR}" ]; then
  echo "[ERROR] FlexWorld not found at ${FLEXWORLD_DIR}"
  exit 1
fi

if [ ! -f "${INPUT_IMAGE}" ]; then
  echo "[ERROR] Input image not found: ${INPUT_IMAGE}"
  exit 1
fi

# ── Environment: flexworld (named env or prefix on scratch) ───────────────
FLEXWORLD_PREFIX="/scratch/network/${USER}/conda_envs/flexworld"
if [ -x "${FLEXWORLD_PREFIX}/bin/python" ]; then
  conda activate "${FLEXWORLD_PREFIX}"
elif conda env list | awk '{print $1}' | grep -qx flexworld; then
  conda activate flexworld
else
  echo "[ERROR] FlexWorld conda env not found. Expected ${FLEXWORLD_PREFIX} or env named flexworld."
  exit 1
fi

mkdir -p "${FLEXWORLD_DIR}/tools/dust3r/checkpoint"
mkdir -p "${FLEXWORLD_DIR}/assets"
mkdir -p "${FLEXWORLD_DIR}/cache"
mkdir -p "${OUTPUT_DIR}"

cp -f "${INPUT_IMAGE}" "${FLEXWORLD_DIR}/assets/${CONFIG_NAME}.png"
echo "[INFO] Copied input to ${FLEXWORLD_DIR}/assets/${CONFIG_NAME}.png"

# Install YAML + orbit patch into FlexWorld checkout
if [ -f "${REPO_ROOT}/install_flexworld_addons.sh" ]; then
  FLEXWORLD_DIR="${FLEXWORLD_DIR}" bash "${REPO_ROOT}/install_flexworld_addons.sh"
else
  echo "[WARN] install_flexworld_addons.sh not found; ensure configs/examples/${CONFIG_NAME}.yaml exists in FlexWorld."
fi

cd "${FLEXWORLD_DIR}"

echo "[INFO] Running main_3dgs.py (multi-traj video + GS training; long-running)..."
echo "[INFO]   Config name: ${CONFIG_NAME}"
echo "[INFO]   Output dir:  ${OUTPUT_DIR}"

python main_3dgs.py --name "${CONFIG_NAME}" --output_dir "${OUTPUT_DIR}"

echo "[INFO] Done at $(date)"
echo "[INFO] Look for: ${OUTPUT_DIR}/${CONFIG_NAME}_gs_final_2.ply"
echo "[INFO]          ${OUTPUT_DIR}/${CONFIG_NAME}_render_novel_view.mp4"
