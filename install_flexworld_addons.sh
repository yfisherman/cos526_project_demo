#!/usr/bin/env bash
# Copy FlexWorld addon configs and apply orbit trajectory patch.
# Usage:
#   FLEXWORLD_DIR=/path/to/FlexWorld bash install_flexworld_addons.sh
# Or from repo root (local FlexWorld clone):
#   bash install_flexworld_addons.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXWORLD_DIR="${FLEXWORLD_DIR:-${ROOT_DIR}/FlexWorld}"

if [ ! -d "${FLEXWORLD_DIR}/ops/utils" ]; then
  echo "[ERROR] FlexWorld not found at ${FLEXWORLD_DIR}"
  echo "        Set FLEXWORLD_DIR or run setup_flexworld.sh to clone FlexWorld under this repo."
  exit 1
fi

mkdir -p "${FLEXWORLD_DIR}/configs/examples"
cp -f "${ROOT_DIR}/flexworld_addons/configs/examples/gemini_painting.yaml" \
  "${FLEXWORLD_DIR}/configs/examples/gemini_painting.yaml"
echo "[OK] Installed configs/examples/gemini_painting.yaml"

# --force shortens an existing long orbit (avoids CogVideo OOM on ~20GB GPU slices).
python3 "${ROOT_DIR}/scripts/apply_flexworld_orbit_traj.py" "${FLEXWORLD_DIR}" --force

echo "[OK] FlexWorld addons installed."
