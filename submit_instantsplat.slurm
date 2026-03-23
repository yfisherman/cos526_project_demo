#!/bin/bash
#SBATCH --job-name=instantsplat-fw
#SBATCH --gres=gpu:1
# Adroit: use full A100 80GB (adroit-h11g1), not MIG 20GB on h11g2.
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-instantsplat-%j.out

set -euo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"
REPO_ROOT="${PROJECT_ROOT}"
if [ ! -f "${REPO_ROOT}/install_flexworld_addons.sh" ] && [ -f "${SLURM_SUBMIT_DIR}/install_flexworld_addons.sh" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

FLEXWORLD_DIR="${PROJECT_ROOT}/FlexWorld"
INSTANTSPLAT_DIR="${PROJECT_ROOT}/InstantSplat"
INPUT_IMAGE="${PROJECT_ROOT}/outputs/destylized/Gemini_Generated_Image_vi7dsfvi7dsfvi7d.png"
FW_BRIDGE_OUT="${PROJECT_ROOT}/outputs/flexworld_instantsplat_bridge"
VIDEO_NAME="bridge_orbit"
SCENE_NAME="flexworld_gemini"
N_VIEWS="${N_VIEWS:-8}"
GS_ITER="${GS_ITER:-1000}"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"

export TORCH_HOME="/scratch/network/${USER}/torch_hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

module purge
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"

echo "[INFO] InstantSplat pipeline at $(date)"

for d in "${FLEXWORLD_DIR}" "${INSTANTSPLAT_DIR}"; do
  if [ ! -d "${d}" ]; then
    echo "[ERROR] Missing ${d}. Run setup_flexworld.sh and bash setup_instantsplat.sh on ${PROJECT_ROOT} first."
    exit 1
  fi
done

if [ ! -f "${INPUT_IMAGE}" ]; then
  echo "[ERROR] Input image not found: ${INPUT_IMAGE}"
  exit 1
fi

FLEXWORLD_PREFIX="/scratch/network/${USER}/conda_envs/flexworld"
INSTANTSPLAT_PREFIX="/scratch/network/${USER}/conda_envs/instantsplat"
if [ ! -x "${FLEXWORLD_PREFIX}/bin/python" ] && ! conda env list | awk '{print $1}' | grep -qx flexworld; then
  echo "[ERROR] FlexWorld conda env not found at ${FLEXWORLD_PREFIX}"
  exit 1
fi
if [ ! -x "${INSTANTSPLAT_PREFIX}/bin/python" ] && ! conda env list | awk '{print $1}' | grep -qx instantsplat; then
  echo "[ERROR] InstantSplat conda env not found at ${INSTANTSPLAT_PREFIX}. Run: bash ${PROJECT_ROOT}/scripts/bootstrap_instantsplat_env.sh"
  exit 1
fi

mkdir -p "${FW_BRIDGE_OUT}" "${FLEXWORLD_DIR}/assets" "${FLEXWORLD_DIR}/cache"
cp -f "${INPUT_IMAGE}" "${FLEXWORLD_DIR}/assets/${VIDEO_NAME}.png"

if [ -f "${REPO_ROOT}/install_flexworld_addons.sh" ]; then
  FLEXWORLD_DIR="${FLEXWORLD_DIR}" bash "${REPO_ROOT}/install_flexworld_addons.sh"
fi

if [ -x "${FLEXWORLD_PREFIX}/bin/python" ]; then conda activate "${FLEXWORLD_PREFIX}"; else conda activate flexworld; fi
cd "${FLEXWORLD_DIR}"

echo "[INFO] FlexWorld video_generate orbit -> ${FW_BRIDGE_OUT}"
python video_generate.py \
  --input_image_path "${FLEXWORLD_DIR}/assets/${VIDEO_NAME}.png" \
  --output_dir "${FW_BRIDGE_OUT}" \
  --traj orbit \
  --name "${VIDEO_NAME}"

MP4="${FW_BRIDGE_OUT}/${VIDEO_NAME}.mp4"
test -f "${MP4}" || { echo "[ERROR] Expected ${MP4}"; exit 1; }

SCENE_ROOT="${INSTANTSPLAT_DIR}/assets/examples/${SCENE_NAME}"
mkdir -p "${SCENE_ROOT}"

if [ -x "${FLEXWORLD_PREFIX}/bin/python" ]; then conda activate "${FLEXWORLD_PREFIX}"; else conda activate flexworld; fi
python "${REPO_ROOT}/scripts/prepare_instantsplat_from_flexworld.py" \
  "${MP4}" \
  "${SCENE_ROOT}" \
  --num_views "${N_VIEWS}"

MODEL_OUT="${PROJECT_ROOT}/outputs/instantsplat/${SCENE_NAME}/${N_VIEWS}_views"
mkdir -p "${MODEL_OUT}"

if [ -x "${INSTANTSPLAT_PREFIX}/bin/python" ]; then conda activate "${INSTANTSPLAT_PREFIX}"; else conda activate instantsplat; fi
bash "${REPO_ROOT}/scripts/run_instantsplat_pipeline.sh" \
  "${INSTANTSPLAT_DIR}" \
  "${SCENE_ROOT}" \
  "${MODEL_OUT}" \
  "${N_VIEWS}" \
  "${GS_ITER}"

echo "[INFO] Done at $(date)"
echo "[INFO] Logs and InstantSplat model under ${MODEL_OUT}"
