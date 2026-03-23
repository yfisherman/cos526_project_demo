#!/usr/bin/env bash
# Check for expected main_3dgs.py outputs. Usage:
#   bash scripts/verify_flexworld_3dgs_outputs.sh <OUTPUT_DIR> [NAME_PREFIX]
# Example:
#   bash scripts/verify_flexworld_3dgs_outputs.sh /scratch/network/$USER/cos526_project/outputs/flexworld_3dgs gemini_painting
set -euo pipefail

OUT="${1:?output dir}"
NAME="${2:-gemini_painting}"

ok=0
for f in \
  "${OUT}/${NAME}_gs_final_2.ply" \
  "${OUT}/${NAME}_render_novel_view.mp4"
do
  if [ -f "$f" ]; then
    echo "[OK] $f ($(du -h "$f" | cut -f1))"
    ok=$((ok + 1))
  else
    echo "[MISSING] $f"
  fi
done

if [ "$ok" -eq 2 ]; then
  echo "[PASS] Both primary artifacts present."
  exit 0
fi
echo "[FAIL] Run submit_flexworld_3dgs.slurm on GPU or check logs."
exit 1
