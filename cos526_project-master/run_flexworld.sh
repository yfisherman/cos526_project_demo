#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXWORLD_DIR="${ROOT_DIR}/FlexWorld"

INPUT_RAW="${1:-${ROOT_DIR}/outputs/destylized}"
INPUT_DIR="$(cd "${INPUT_RAW}" 2>/dev/null && pwd)" || {
  echo "[ERROR] Input directory not found: ${INPUT_RAW}"
  exit 1
}

OUTPUT_RAW="${2:-${ROOT_DIR}/outputs/flexworld}"
mkdir -p "${OUTPUT_RAW}"
OUTPUT_DIR="$(cd "${OUTPUT_RAW}" && pwd)"

# Trajectory: backward, forward, orbit, rotate_left, ... or path to a reference video (MASt3R poses).
TRAJ="${3:-backward}"

if [ ! -d "${FLEXWORLD_DIR}" ]; then
  echo "[ERROR] FlexWorld repo not found at ${FLEXWORLD_DIR}"
  echo "[ERROR] Run setup_flexworld.sh first."
  exit 1
fi

shopt -s nullglob
IMAGES=( "${INPUT_DIR}"/*.png "${INPUT_DIR}"/*.jpg "${INPUT_DIR}"/*.jpeg )
if [ ${#IMAGES[@]} -eq 0 ]; then
  echo "[ERROR] No input images found in ${INPUT_DIR}"
  exit 1
fi

echo "[INFO] FlexWorld video generation"
echo "[INFO]   Input dir : ${INPUT_DIR}"
echo "[INFO]   Output dir: ${OUTPUT_DIR}"
echo "[INFO]   Trajectory: ${TRAJ}"
echo "[INFO]   Images    : ${#IMAGES[@]}"

cd "${FLEXWORLD_DIR}"

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

  echo ""
  echo "[INFO] ──────────────────────────────────────────"
  echo "[INFO] Processing: ${base}"
  echo "[INFO] Output to:  ${sample_out}"

  python video_generate.py \
    --input_image_path "${img}" \
    --output_dir "${sample_out}" \
    --traj "${TRAJ}" \
    --name "${stem}"
done

echo ""
echo "[INFO] FlexWorld generation complete. Outputs in ${OUTPUT_DIR}"
