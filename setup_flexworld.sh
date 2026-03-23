#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[INFO] Using project root: ${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Load Anaconda first (e.g., module load anaconda3/2024.2)."
  exit 1
fi

eval "$(conda shell.bash hook)"

# ── 1. Clone FlexWorld ───────────────────────────────────────────────
FLEXWORLD_DIR="${ROOT_DIR}/FlexWorld"
if [ ! -d "${FLEXWORLD_DIR}/.git" ]; then
  echo "[INFO] Cloning FlexWorld repository..."
  git clone https://github.com/ML-GSAI/FlexWorld.git
else
  echo "[INFO] FlexWorld repo already exists, skipping clone."
fi

# ── 2. Create conda environment ─────────────────────────────────────
ENV_NAME="flexworld"
echo "[INFO] Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.11

conda activate "${ENV_NAME}"

conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2

pip install -r "${FLEXWORLD_DIR}/requirements.txt"

cd "${FLEXWORLD_DIR}/tools/CogVideo"
pip install -r requirements.txt
cd "${ROOT_DIR}"

pip install basicsr

conda deactivate

# ── 3. Download pretrained models ────────────────────────────────────
echo "[INFO] Downloading pretrained models..."

# DUSt3R and MASt3R checkpoints
# Config basic.yaml uses "checkpoint" (singular) as path
mkdir -p "${FLEXWORLD_DIR}/tools/dust3r/checkpoint"
if [ ! -f "${FLEXWORLD_DIR}/tools/dust3r/checkpoint/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
  echo "[INFO] Downloading DUSt3R checkpoint..."
  wget -q --show-progress \
    https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -O "${FLEXWORLD_DIR}/tools/dust3r/checkpoint/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
fi

if [ ! -f "${FLEXWORLD_DIR}/tools/dust3r/checkpoint/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" ]; then
  echo "[INFO] Downloading MASt3R checkpoint..."
  wget -q --show-progress \
    https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    -O "${FLEXWORLD_DIR}/tools/dust3r/checkpoint/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
fi

# Also create "checkpoints" symlink in case any code references the plural form
if [ ! -e "${FLEXWORLD_DIR}/tools/dust3r/checkpoints" ]; then
  ln -s checkpoint "${FLEXWORLD_DIR}/tools/dust3r/checkpoints"
fi

# CogVideoX-SAT (from HuggingFace)
echo "[INFO] Downloading CogVideoX-SAT checkpoints from HuggingFace..."
conda activate "${ENV_NAME}"
pip install -U huggingface_hub
huggingface-cli download GSAI-ML/FlexWorld --local-dir "${FLEXWORLD_DIR}/tools/CogVideo/checkpoints"
conda deactivate

# Real-ESRGAN
mkdir -p "${FLEXWORLD_DIR}/tools/Real_ESRGAN/weights"
if [ ! -f "${FLEXWORLD_DIR}/tools/Real_ESRGAN/weights/RealESRGAN_x4plus.pth" ]; then
  echo "[INFO] Downloading Real-ESRGAN weights..."
  wget -q --show-progress \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O "${FLEXWORLD_DIR}/tools/Real_ESRGAN/weights/RealESRGAN_x4plus.pth"
fi

echo "[INFO] FlexWorld setup complete."

if [ -f "${ROOT_DIR}/install_flexworld_addons.sh" ]; then
  echo "[INFO] Installing project addons (gemini_painting.yaml, orbit trajectory patch)..."
  FLEXWORLD_DIR="${FLEXWORLD_DIR}" bash "${ROOT_DIR}/install_flexworld_addons.sh" || true
fi

echo "[INFO] To run: bash run_flexworld.sh [input_dir] [output_dir] [trajectory]"
echo "[INFO] For trained 3DGS: sbatch submit_flexworld_3dgs.slurm (see docs/GAUSSIAN_SPLAT_PIPELINE.md)"
