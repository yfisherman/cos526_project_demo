#!/usr/bin/env bash
# Clone InstantSplat and symlink MASt3R weights from FlexWorld (avoid re-download).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANTSPLAT_DIR="${INSTANTSPLAT_DIR:-${ROOT_DIR}/InstantSplat}"
FLEXWORLD_DIR="${FLEXWORLD_DIR:-${ROOT_DIR}/FlexWorld}"

echo "[INFO] InstantSplat dir: ${INSTANTSPLAT_DIR}"
echo "[INFO] FlexWorld dir (for MASt3R ckpt): ${FLEXWORLD_DIR}"

if [ ! -d "${INSTANTSPLAT_DIR}/.git" ]; then
  echo "[INFO] Cloning NVlabs/InstantSplat..."
  git clone --recursive https://github.com/NVlabs/InstantSplat.git "${INSTANTSPLAT_DIR}"
else
  echo "[INFO] InstantSplat repo already exists."
fi

MAST3R_NAME="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SRC="${FLEXWORLD_DIR}/tools/dust3r/checkpoint/${MAST3R_NAME}"
DST_DIR="${INSTANTSPLAT_DIR}/mast3r/checkpoints"
mkdir -p "${DST_DIR}"
DST="${DST_DIR}/${MAST3R_NAME}"

if [ -f "${SRC}" ]; then
  if [ -e "${DST}" ] && [ ! -L "${DST}" ]; then
    echo "[INFO] Keeping existing file at ${DST}"
  else
    ln -sf "${SRC}" "${DST}"
    echo "[OK] Symlinked MASt3R checkpoint: ${DST} -> ${SRC}"
  fi
else
  echo "[WARN] MASt3R checkpoint not at ${SRC}"
  echo "       Download to FlexWorld (see setup_flexworld.sh) or:"
  echo "       wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/${MAST3R_NAME} -P ${DST_DIR}"
fi

cat <<EOF

[INFO] Next: create conda env (CUDA 12.1 example; adjust for your cluster):

  conda create -n instantsplat python=3.10.13 cmake=3.14.0 -y
  conda activate instantsplat
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  cd ${INSTANTSPLAT_DIR}
  pip install -r requirements.txt
  pip install submodules/simple-knn
  pip install submodules/diff-gaussian-rasterization
  pip install submodules/fused-ssim

Optional (DUSt3R RoPE speed): cd croco/models/curope && python setup.py build_ext --inplace

EOF
