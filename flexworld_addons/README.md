# FlexWorld addons (this repo)

- **`configs/examples/gemini_painting.yaml`** — `main_3dgs.py` config; input image must be at `FlexWorld/assets/gemini_painting.png` (Slurm script copies it).
- **Orbit trajectory** — applied by `../scripts/apply_flexworld_orbit_traj.py` when you run `install_flexworld_addons.sh`.

Install into a FlexWorld clone:

```bash
export FLEXWORLD_DIR=/path/to/FlexWorld
bash install_flexworld_addons.sh
```
