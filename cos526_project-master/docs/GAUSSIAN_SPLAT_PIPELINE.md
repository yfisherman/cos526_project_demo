# Gaussian splat pipeline (FlexWorld)

Two paths improve on the raw `*_pcd_ori.ply` from `video_generate.py`:

## 0. Adroit / offline compute nodes

GPU nodes often have **no outbound DNS**. `main_3dgs.py` uses LPIPS, which triggers a **torchvision AlexNet** download on first use unless weights are cached.

**One-time (on a login node with network):**

```bash
module load anaconda3/2024.2
eval "$(conda shell.bash hook)"
conda activate /scratch/network/$USER/conda_envs/flexworld   # or: conda activate flexworld
export TORCH_HOME=/scratch/network/$USER/torch_hub
mkdir -p "$TORCH_HOME"
python -c "import torchvision.models as m; m.alexnet(weights='DEFAULT')"
```

The Slurm scripts set `export TORCH_HOME=/scratch/network/$USER/torch_hub` so jobs use this cache.

### basicsr / Real-ESRGAN vs torchvision

If `main_3dgs.py` crashes with `No module named 'torchvision.transforms.functional_tensor'`, that comes from **basicsr** (pulled in by FlexWorld’s Real-ESRGAN wrapper). Newer **torchvision** removed that module. The bundled `gemini_painting.yaml` sets **`sr: False`** so training skips `SRTool` and avoids basicsr on compute nodes.

### CogVideo OOM on smaller GPU partitions

**Adroit:** `--constraint=a100` often schedules **adroit-h11g2** (A100 **MIG 20 GB** slices). CogVideo+VAE then OOMs. The Slurm scripts use **`--constraint=gpu80`**, which targets **adroit-h11g1** (4× **A100 80 GB**). Verify with `sinfo -N -o '%N %G %f'`.

If you still OOM, shorten the `orbit` trajectory and re-run `python scripts/apply_flexworld_orbit_traj.py "$FLEXWORLD_DIR" --force`.

---

## 1. FlexWorld `main_3dgs.py` (recommended first)

Trains 3D Gaussians inside FlexWorld (multi-trajectory CogVideo + gsplat + refinement).

**One-time:** install addons into your FlexWorld checkout (YAML + `orbit` traj for other scripts):

```bash
export FLEXWORLD_DIR=/scratch/network/$USER/cos526_project/FlexWorld
bash install_flexworld_addons.sh
```

**Run on GPU (Slurm):**

```bash
cd /scratch/network/$USER/cos526_project   # or your repo root with these scripts
sbatch submit_flexworld_3dgs.slurm
```

**Outputs (typical):**

- `outputs/flexworld_3dgs/gemini_painting_gs_final_2.ply` — optimized Gaussian splat
- `outputs/flexworld_3dgs/gemini_painting_render_novel_view.mp4`

**Customize:** edit `flexworld_addons/configs/examples/gemini_painting.yaml` (prompt + name), re-run `install_flexworld_addons.sh`, and set `CONFIG_NAME` / asset copy in `submit_flexworld_3dgs.slurm` if you rename.

## 2. InstantSplat on FlexWorld video (comparison / sparse-view GS)

Uses MASt3R-style initialization inside InstantSplat; needs **multiple views** → use FlexWorld video with **`--traj orbit`**.

**Setup:**

```bash
bash setup_instantsplat.sh
# Then create conda env `instantsplat` per the printed instructions.
```

**Run end-to-end (Slurm):**

```bash
sbatch submit_instantsplat.slurm
```

Optional env overrides on the `sbatch` line:

```bash
N_VIEWS=10 GS_ITER=1500 sbatch submit_instantsplat.slurm
```

**Manual steps (debug):**

```bash
conda activate flexworld
cd $FLEXWORLD_DIR
python video_generate.py --input_image_path ... --output_dir ... --traj orbit --name myrun

conda activate flexworld
python scripts/prepare_instantsplat_from_flexworld.py myrun.mp4 $INSTANTSPLAT_DIR/assets/examples/my_scene --num_views 8

conda activate instantsplat
bash scripts/run_instantsplat_pipeline.sh $INSTANTSPLAT_DIR $INSTANTSPLAT_DIR/assets/examples/my_scene $OUT_MODEL_DIR 8 1000
```

## Comparing results

| Artifact | Method | What it is |
|----------|--------|------------|
| `*_pcd_ori.ply` | `video_generate.py` | DUSt3R point cloud only; not trained 3DGS |
| `*_gs_final_2.ply` | `main_3dgs.py` | Trained Gaussians (FlexWorld / gsplat) |
| InstantSplat `point_cloud/` etc. | InstantSplat | Trained Gaussians + pose opt (see InstantSplat `output_*` layout) |

Viewers: FlexWorld `3dga_viewer.py` (per upstream README), or load PLY in your preferred 3DGS viewer.

## Verify after a 3DGS job

```bash
bash scripts/verify_flexworld_3dgs_outputs.sh /path/to/outputs/flexworld_3dgs gemini_painting
```
