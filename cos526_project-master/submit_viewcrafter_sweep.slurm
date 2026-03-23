#!/bin/bash
#SBATCH --job-name=vc-sweep
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-vc-sweep-%j.out

set -eo pipefail

PROJECT_ROOT="/scratch/network/${USER}/cos526_project"
VIEWCRAFTER_DIR="${PROJECT_ROOT}/ViewCrafter"
IMAGE="/scratch/network/${USER}/cos526_project/outputs/destylized/Gemini_Generated_Image_vi7dsfvi7dsfvi7d.png"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/viewcrafter_sweep"

export HF_HOME="/scratch/network/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"

module purge
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"
conda activate viewcrafter

cd "${VIEWCRAFTER_DIR}"

run_viewcrafter() {
  local name="$1"
  shift
  local out_dir="${OUTPUT_BASE}/${name}"
  mkdir -p "${out_dir}"
  echo ""
  echo "============================================"
  echo "[INFO] Run: ${name}"
  echo "[INFO] Args: $@"
  echo "[INFO] Start: $(date)"
  echo "============================================"
  python inference.py \
    --image_dir "${IMAGE}" \
    --out_dir "${out_dir}" \
    --ckpt_path "./checkpoints/model.ckpt" \
    --config "configs/inference_pvd_1024.yaml" \
    --model_path "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --device "cuda:0" \
    --height 576 \
    --width 1024 \
    --video_length 25 \
    --ddim_steps 50 \
    --mode "single_view_txt" \
    "$@"
  echo "[INFO] Done: ${name} at $(date)"
}

echo "[INFO] Starting ViewCrafter sweep at $(date)"
echo "[INFO] Image: ${IMAGE}"

# Run 1: Baseline (your original params)
run_viewcrafter "01_baseline" \
  --traj_txt "test/trajs/loop2.txt" \
  --elevation 5 \
  --center_scale 1.0 \
  --seed 123

# Run 2: elevation=0 (painting viewed head-on)
run_viewcrafter "02_elev0" \
  --traj_txt "test/trajs/loop2.txt" \
  --elevation 0 \
  --center_scale 1.0 \
  --seed 123

# Run 3: eta=0 (deterministic/sharper)
run_viewcrafter "03_eta0" \
  --traj_txt "test/trajs/loop2.txt" \
  --elevation 0 \
  --center_scale 1.0 \
  --seed 123 \
  --ddim_eta 0.0

# Run 4: gentle trajectory + different seed
run_viewcrafter "04_gentle_seed42" \
  --traj_txt "test/trajs/gentle_loop.txt" \
  --elevation 0 \
  --center_scale 0.9 \
  --seed 42

# Run 5: loop1 trajectory (orbits opposite direction)
run_viewcrafter "05_loop1_seed42" \
  --traj_txt "test/trajs/loop1.txt" \
  --elevation 0 \
  --center_scale 1.0 \
  --seed 42

# Run 6: wave trajectory (lateral movement)
run_viewcrafter "06_wave_seed77" \
  --traj_txt "test/trajs/wave1.txt" \
  --elevation 0 \
  --center_scale 1.0 \
  --seed 77

echo ""
echo "============================================"
echo "[INFO] All sweep runs complete at $(date)"
echo "[INFO] Outputs in: ${OUTPUT_BASE}"
echo "============================================"
