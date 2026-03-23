#!/usr/bin/env bash
set -euo pipefail
# Run InstantSplat init_geo, train, render. Args: ISP_ROOT SOURCE_PATH MODEL_OUT [N_VIEWS] [ITER]

ISP_ROOT="$(cd "$1" && pwd)"
SOURCE_PATH="$(cd "$2" && pwd)"
mkdir -p "$3"
MODEL_PATH="$(cd "$3" && pwd)"
N_VIEWS="${4:-8}"
ITER="${5:-1000}"

test -f "${ISP_ROOT}/init_geo.py" || { echo "missing init_geo.py"; exit 1; }
test -d "${SOURCE_PATH}/images" || { echo "missing images/"; exit 1; }

mkdir -p "${MODEL_PATH}"
cd "${ISP_ROOT}"

python -W ignore ./init_geo.py -s "${SOURCE_PATH}" -m "${MODEL_PATH}" \
  --n_views "${N_VIEWS}" --focal_avg --co_vis_dsp --conf_aware_ranking --infer_video \
  2>&1 | tee "${MODEL_PATH}/01_init_geo.log"

python ./train.py -s "${SOURCE_PATH}" -m "${MODEL_PATH}" -r 1 \
  --n_views "${N_VIEWS}" --iterations "${ITER}" --pp_optimizer --optim_pose \
  2>&1 | tee "${MODEL_PATH}/02_train.log"

python ./render.py -s "${SOURCE_PATH}" -m "${MODEL_PATH}" -r 1 \
  --n_views "${N_VIEWS}" --iterations "${ITER}" --infer_video \
  2>&1 | tee "${MODEL_PATH}/03_render.log"

echo "OK: ${MODEL_PATH}"
