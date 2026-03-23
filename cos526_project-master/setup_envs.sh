#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[INFO] Using project root: ${ROOT_DIR}"

# Ensure conda is available in non-interactive shells.
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Load Anaconda first (e.g., module load anaconda3/2024.2)."
  exit 1
fi

eval "$(conda shell.bash hook)"

mkdir -p inputs/paintings outputs/destylized outputs/depth_maps outputs/viewcrafter

echo "[INFO] Creating environment: cos526_destylize"
conda create -y -n cos526_destylize python=3.10
conda activate cos526_destylize

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors pillow numpy opencv-python controlnet_aux

conda deactivate

echo "[INFO] Setting up ViewCrafter repository"
if [ ! -d "ViewCrafter/.git" ]; then
  git clone https://github.com/Drexubery/ViewCrafter.git
else
  echo "[INFO] ViewCrafter repo already exists, skipping clone."
fi

echo "[INFO] Creating environment: viewcrafter"
conda create -y -n viewcrafter python=3.9.16
conda activate viewcrafter

pip install --upgrade pip
pip install -r ViewCrafter/requirements.txt
conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

mkdir -p ViewCrafter/checkpoints
wget -nc https://huggingface.co/Drexubery/ViewCrafter_25/resolve/main/model.ckpt -P ViewCrafter/checkpoints/
wget -nc https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P ViewCrafter/checkpoints/

conda deactivate

echo "[INFO] Environment setup complete."
