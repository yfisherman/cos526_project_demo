#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWCRAFTER_DIR="${ROOT_DIR}/ViewCrafter"
CKPT_DIR="${VIEWCRAFTER_DIR}/checkpoints"

INPUT_RAW="${1:-${ROOT_DIR}/outputs/destylized}"
INPUT_DIR="$(cd "${INPUT_RAW}" 2>/dev/null && pwd)" || {
  echo "[ERROR] Input directory not found: ${INPUT_RAW}"
  exit 1
}

OUTPUT_RAW="${2:-${ROOT_DIR}/outputs/viewcrafter}"
mkdir -p "${OUTPUT_RAW}"
OUTPUT_DIR="$(cd "${OUTPUT_RAW}" && pwd)"

TRAJ_TXT="${3:-test/trajs/loop2.txt}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${VIEWCRAFTER_DIR}" ]; then
  echo "[ERROR] ViewCrafter repo not found at ${VIEWCRAFTER_DIR}"
  echo "[ERROR] Run setup_envs.sh first."
  exit 1
fi

mkdir -p "${CKPT_DIR}"

if [ ! -f "${CKPT_DIR}/model.ckpt" ]; then
  echo "[INFO] Downloading ViewCrafter_25 checkpoint..."
  wget -q --show-progress https://huggingface.co/Drexubery/ViewCrafter_25/resolve/main/model.ckpt -O "${CKPT_DIR}/model.ckpt"
fi

if [ ! -f "${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
  echo "[INFO] Downloading DUSt3R checkpoint..."
  wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -O "${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
fi

shopt -s nullglob
IMAGES=( "${INPUT_DIR}"/*.png "${INPUT_DIR}"/*.jpg "${INPUT_DIR}"/*.jpeg )
if [ ${#IMAGES[@]} -eq 0 ]; then
  echo "[ERROR] No input images found in ${INPUT_DIR}"
  exit 1
fi

cd "${VIEWCRAFTER_DIR}"

for img in "${IMAGES[@]}"; do
  img="$(realpath "${img}")"
  if [ ! -f "${img}" ]; then
    echo "[WARN] Skipping missing file: ${img}"
    continue
  fi
  base="$(basename "${img}")"
  stem="${base%.*}"
  sample_out="${OUTPUT_DIR}/${stem}"
  mkdir -p "${sample_out}"

  echo "[INFO] Running ViewCrafter for ${base}"
  echo "[INFO] Image path: ${img}"
  python inference.py \
    --image_dir "${img}" \
    --out_dir "${sample_out}" \
    --traj_txt "${TRAJ_TXT}" \
    --mode "single_view_txt" \
    --center_scale 1.0 \
    --elevation 5 \
    --seed 123 \
    --d_theta -30 \
    --d_phi 45 \
    --d_r -0.2 \
    --d_x 50 \
    --d_y 25 \
    --ckpt_path "./checkpoints/model.ckpt" \
    --config "configs/inference_pvd_1024.yaml" \
    --ddim_steps 50 \
    --video_length 25 \
    --device "cuda:0" \
    --height 576 \
    --width 1024 \
    --model_path "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
done

echo "[INFO] ViewCrafter generation complete. Outputs in ${OUTPUT_DIR}"
