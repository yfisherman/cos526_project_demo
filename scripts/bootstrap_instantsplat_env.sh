#!/usr/bin/env bash
# Create InstantSplat conda env at /scratch/network/$USER/conda_envs/instantsplat and pip-install deps.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/network/${USER}/cos526_project}"
INSTANTSPLAT_DIR="${INSTANTSPLAT_DIR:-${PROJECT_ROOT}/InstantSplat}"
PREFIX="/scratch/network/${USER}/conda_envs/instantsplat"

module purge
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"

if [ ! -d "${INSTANTSPLAT_DIR}" ]; then
  echo "[ERROR] InstantSplat not found at ${INSTANTSPLAT_DIR}. Run setup_instantsplat.sh first."
  exit 1
fi

if [ -x "${PREFIX}/bin/python" ]; then
  echo "[INFO] Env already exists: ${PREFIX}"
  conda activate "${PREFIX}"
else
  echo "[INFO] Creating conda env at ${PREFIX} ..."
  conda create -y -p "${PREFIX}" python=3.10 cmake=3.14 -c conda-forge
  conda activate "${PREFIX}"
  conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
fi

cd "${INSTANTSPLAT_DIR}"
pip install --upgrade pip
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim

echo "[OK] InstantSplat env ready: conda activate ${PREFIX}"
