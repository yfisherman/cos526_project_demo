# COS 526 Project — Painting-to-3D Scene Pipeline

Converts stylized paintings into explorable 3D Gaussian splat scenes using:

1. **Destylization** — SDXL + ControlNet depth img2img to convert a painting into a photorealistic image
2. **FlexWorld** — generates multi-trajectory videos and trains 3D Gaussians from a single image
3. *(Optional)* **InstantSplat** — alternative sparse-view Gaussian splat from FlexWorld video frames

The pipeline was developed and tested on Princeton's [Adroit](https://researchcomputing.princeton.edu/systems/adroit) HPC cluster (A100 80 GB GPUs, Slurm scheduler). All Slurm scripts use `$USER` so they work for any user who clones to `/scratch/network/$USER/cos526_project`.

---

## Repository layout

```
.
├── inputs/paintings/          # Your input painting images (gitignored; add your own)
├── outputs/                   # All generated outputs (gitignored)
├── FlexWorld/                 # FlexWorld clone (gitignored; created by setup_flexworld.sh)
│
├── destylize.py               # Step 1: SDXL+ControlNet destylization
├── run_destylize_sweep.sh     # Run destylize with two hyperparameter sets
│
├── setup_flexworld.sh         # Step 2a: clone FlexWorld, create conda env, download models
├── run_flexworld.sh           # Step 2b: run FlexWorld video_generate.py on a directory
├── install_flexworld_addons.sh# Copy project configs + orbit patch into FlexWorld checkout
├── flexworld_addons/          # Project-specific FlexWorld configs and patches
│   └── configs/examples/
│       └── gemini_painting.yaml  # main_3dgs.py config (sr:False, orbit traj, custom prompt)
│
├── submit_flexworld_3dgs.slurm    # Slurm: full pipeline (video + 3DGS training) ← main job
├── submit_flexworld_single.slurm  # Slurm: video_generate.py only (single image)
├── submit_flexworld.slurm         # Slurm: video_generate.py sweep
│
├── setup_instantsplat.sh      # Optional: set up InstantSplat
├── submit_instantsplat.slurm  # Optional Slurm: InstantSplat on FlexWorld video
│
├── scripts/                   # Helper scripts
│   ├── apply_flexworld_orbit_traj.py      # Patches orbit trajectory length in FlexWorld
│   ├── prepare_instantsplat_from_flexworld.py  # Extracts frames from FlexWorld video
│   ├── run_instantsplat_pipeline.sh       # Runs InstantSplat end-to-end
│   ├── bootstrap_instantsplat_env.sh      # Creates InstantSplat conda env
│   └── verify_flexworld_3dgs_outputs.sh   # Sanity-checks 3DGS job outputs
│
└── docs/
    └── GAUSSIAN_SPLAT_PIPELINE.md  # Detailed notes on the 3DGS pipeline and gotchas
```

---

## Prerequisites

- Linux with CUDA 12.1-compatible GPU (≥ 40 GB VRAM recommended; 80 GB for CogVideo)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- `git`, `wget`
- On Adroit: `module load anaconda3/2024.2` gives you conda

---

## Step 0 — Clone this repo

```bash
git clone https://github.com/rhiaan23/cos526_project.git
cd cos526_project
```

> **Adroit users:** clone to `/scratch/network/$USER/cos526_project` — the Slurm scripts expect this path.

---

## Step 1 — Add your input paintings

Place your painting images (`.jpg`, `.png`) in `inputs/paintings/`:

```bash
mkdir -p inputs/paintings
cp /path/to/your/painting.jpg inputs/paintings/
```

---

## Step 2 — Destylize paintings → photorealistic images

Destylization uses SDXL + ControlNet depth to convert paintings into photorealistic images that FlexWorld can process effectively.

**Install dependencies** (into any Python ≥ 3.10 env with PyTorch + CUDA):

```bash
pip install diffusers controlnet_aux transformers accelerate torch torchvision Pillow
```

**Run on a single image or a directory:**

```bash
python destylize.py \
  --input_dir  inputs/paintings \
  --output_dir outputs/destylized
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--strength` | `0.4` | Img2Img denoise strength; lower = more faithful to painting structure |
| `--guidance_scale` | `5.0` | CFG scale |
| `--controlnet_conditioning_scale` | `0.9` | How strongly depth guides generation |
| `--resize_mode` | `pad` | `pad` (letterbox) or `crop` |
| `--seed` | `42` | Reproducibility |

**Or run a two-setting hyperparameter sweep:**

```bash
bash run_destylize_sweep.sh inputs/paintings outputs/sweeps
```

Outputs land in `outputs/destylized/` (and `outputs/depth_maps/`, `outputs/processed_inputs/` for debug).

**On Adroit (Slurm):**

```bash
sbatch submit_pipeline.slurm
```

---

## Step 3 — Set up FlexWorld

This clones [FlexWorld](https://github.com/ML-GSAI/FlexWorld), creates the `flexworld` conda environment, downloads all pretrained model weights (DUSt3R, MASt3R, CogVideoX-SAT, Real-ESRGAN), and installs project-specific configs.

```bash
bash setup_flexworld.sh
```

This takes a while (model downloads are several GB). It is safe to re-run — existing files are skipped.

**What gets downloaded:**

| Model | Size | Purpose |
|-------|------|---------|
| DUSt3R ViT-Large | ~1.1 GB | Point cloud from video |
| MASt3R ViT-Large | ~1.1 GB | Camera pose estimation |
| CogVideoX-SAT (FlexWorld) | ~15 GB | Video generation |
| Real-ESRGAN x4plus | ~67 MB | Super-resolution (disabled in our config) |

---

## Step 4 — Run FlexWorld (video + 3D Gaussian splat)

### Recommended: full 3DGS pipeline via Slurm

```bash
sbatch submit_flexworld_3dgs.slurm
```

This runs `main_3dgs.py` which:
1. Generates multi-trajectory videos using CogVideo
2. Reconstructs a point cloud with DUSt3R
3. Trains 3D Gaussians with gsplat
4. Renders a novel-view video

**Expected outputs** in `outputs/flexworld_3dgs/`:

```
gemini_painting_gs_final_2.ply        ← trained Gaussian splat (open in any 3DGS viewer)
gemini_painting_render_novel_view.mp4 ← rendered novel-view video
```

**To use your own image** instead of the default `gemini_painting`:

1. Edit `submit_flexworld_3dgs.slurm` — change `INPUT_IMAGE` and `CONFIG_NAME`
2. Edit (or copy) `flexworld_addons/configs/examples/gemini_painting.yaml` — update `name`, `input_image_path`, and the CogVideo `prompt`
3. Re-run `bash install_flexworld_addons.sh` to push the config into the FlexWorld checkout
4. `sbatch submit_flexworld_3dgs.slurm`

### Video only (no 3DGS training)

```bash
# Single image, backward trajectory
sbatch submit_flexworld_single.slurm

# Or run directly (requires conda env active):
conda activate flexworld
bash run_flexworld.sh inputs/paintings outputs/flexworld backward
```

Available trajectories: `backward`, `forward`, `orbit`, `rotate_left`, `rotate_right`, or a path to a reference video for MASt3R-guided poses.

---

## Step 5 — View results

Open the `.ply` file in any 3D Gaussian splat viewer, e.g.:

- [SuperSplat](https://playcanvas.com/supersplat/editor) (browser, no install)
- [3D Gaussian Splat Viewer](https://antimatter15.com/splat/) (browser)
- FlexWorld's own viewer: `python FlexWorld/3dgs_viewer.py`

---

## Optional: InstantSplat (alternative sparse-view 3DGS)

InstantSplat provides an alternative Gaussian splat reconstruction using MASt3R pose initialization on frames extracted from a FlexWorld orbit video.

```bash
# Setup (one time)
bash setup_instantsplat.sh

# Run (Slurm)
sbatch submit_instantsplat.slurm

# Override defaults
N_VIEWS=10 GS_ITER=1500 sbatch submit_instantsplat.slurm
```

See `docs/GAUSSIAN_SPLAT_PIPELINE.md` for a detailed comparison of FlexWorld 3DGS vs InstantSplat outputs.

---

## Troubleshooting

**CogVideo OOM on smaller GPU partitions**
The Slurm scripts use `--constraint=gpu80` to target full A100 80 GB cards. On Adroit, bare `--constraint=a100` can land on MIG 20 GB slices which OOM. If you still OOM, shorten the orbit trajectory:
```bash
python scripts/apply_flexworld_orbit_traj.py FlexWorld --force
```

**`No module named 'torchvision.transforms.functional_tensor'`**
This is a basicsr/Real-ESRGAN incompatibility with newer torchvision. The bundled `gemini_painting.yaml` sets `sr: False` to skip it. If you see this in a custom config, add `sr: False`.

**AlexNet / LPIPS download fails on compute node (no internet)**
Pre-cache on the login node before submitting:
```bash
module load anaconda3/2024.2 && conda activate flexworld
export TORCH_HOME=/scratch/network/$USER/torch_hub
python -c "import torchvision.models as m; m.alexnet(weights='DEFAULT')"
```

---

## References

- [FlexWorld](https://github.com/ML-GSAI/FlexWorld) — ML-GSAI
- [InstantSplat](https://github.com/NVlabs/InstantSplat) — NVIDIA
- [DUSt3R](https://github.com/naver/dust3r) / [MASt3R](https://github.com/naver/mast3r) — Naver Labs
- [CogVideoX](https://github.com/THUDM/CogVideo) — THUDM
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) + [ControlNet Depth](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0)
