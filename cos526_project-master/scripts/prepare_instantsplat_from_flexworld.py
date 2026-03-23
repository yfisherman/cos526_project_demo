#!/usr/bin/env python3
"""
Extract evenly spaced frames from a FlexWorld MP4 into an InstantSplat scene layout:

  <out_root>/<scene_name>/images/*.png

Usage:
  python scripts/prepare_instantsplat_from_flexworld.py \\
    outputs/flexworld/foo/video.mp4 \\
    /path/to/InstantSplat/assets/examples/my_scene \\
    --num_views 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="FlexWorld video -> InstantSplat images/")
    ap.add_argument("video", type=Path, help="Path to FlexWorld output .mp4")
    ap.add_argument(
        "scene_dir",
        type=Path,
        help="InstantSplat scene root (we create images/ under it)",
    )
    ap.add_argument(
        "--num_views",
        type=int,
        default=8,
        help="Number of frames to sample (6–12 typical for InstantSplat)",
    )
    ap.add_argument(
        "--prefix",
        default="frame",
        help="Output filenames: {prefix}_0000.png ...",
    )
    args = ap.parse_args()

    vid = args.video.resolve()
    if not vid.is_file():
        print(f"[ERROR] Video not found: {vid}", file=sys.stderr)
        return 1

    scene = args.scene_dir.resolve()
    images_dir = scene / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading {vid} ...")
    frames = imageio.mimread(str(vid), memtest=False)
    n = len(frames)
    if n == 0:
        print("[ERROR] No frames decoded from video.", file=sys.stderr)
        return 1

    k = min(args.num_views, n)
    idxs = np.linspace(0, n - 1, k, dtype=int)
    print(f"[INFO] Saving {k} frames from {n} total -> {images_dir}")

    for j, i in enumerate(idxs):
        arr = np.asarray(frames[int(i)])
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        out = images_dir / f"{args.prefix}_{j:04d}.png"
        imageio.imwrite(str(out), arr)

    print(f"[OK] Wrote {k} images under {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
