#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${PROJECT_ROOT}/inputs/paintings}"
SWEEP_ROOT="${2:-${PROJECT_ROOT}/outputs/sweeps}"

mkdir -p "${SWEEP_ROOT}"

echo "[INFO] Running Destylize Sweep A"
python "${PROJECT_ROOT}/destylize.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${SWEEP_ROOT}/setA_destylized" \
  --depth_dir "${SWEEP_ROOT}/setA_depth_maps" \
  --processed_dir "${SWEEP_ROOT}/setA_processed_inputs" \
  --seed 42 \
  --strength 0.40 \
  --guidance_scale 5.0 \
  --controlnet_conditioning_scale 0.9 \
  --num_inference_steps 40 \
  --resize_mode pad

echo "[INFO] Running Destylize Sweep B"
python "${PROJECT_ROOT}/destylize.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${SWEEP_ROOT}/setB_destylized" \
  --depth_dir "${SWEEP_ROOT}/setB_depth_maps" \
  --processed_dir "${SWEEP_ROOT}/setB_processed_inputs" \
  --seed 42 \
  --strength 0.30 \
  --guidance_scale 4.5 \
  --controlnet_conditioning_scale 1.0 \
  --num_inference_steps 40 \
  --resize_mode pad

echo "[INFO] Sweep complete at ${SWEEP_ROOT}"
