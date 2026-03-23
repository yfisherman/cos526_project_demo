#!/usr/bin/env python3
"""
Idempotently add the 'orbit' trajectory to FlexWorld ops/utils/all_traj.py.
Usage: python scripts/apply_flexworld_orbit_traj.py /path/to/FlexWorld
"""
from __future__ import annotations

import pathlib
import sys

# Shorter orbit (~25 poses) to fit CogVideo+VAE on ~20GB GPUs (e.g. MIG A100); was 16+32 -> ~49 frames OOM.
ORBIT_BRANCH = """    elif move_instruct == "orbit":
        # Forward dolly then ~half orbit for more orthogonal views (splats / MASt3R).
        traj = (
            CamPlanner()
            .add_traj()
            .move_forward(0.1, num_frames=8)
            .move_orbit_to(0, 179.999, 0.1, 0, num_frames=16)
            .finish()
        )"""

ORBIT_BRANCH_LEGACY = """    elif move_instruct == "orbit":
        # Forward dolly then ~half orbit for more orthogonal views (splats / MASt3R).
        traj = (
            CamPlanner()
            .add_traj()
            .move_forward(0.1, num_frames=16)
            .move_orbit_to(0, 179.999, 0.1, 0, num_frames=32)
            .finish()
        )"""

VALID_REPLACEMENTS = [
    (
        'valid_move_instructs = ["up","down","left","right","forward","backward","rotate_left","rotate_right"]',
        'valid_move_instructs = ["up","down","left","right","forward","backward","rotate_left","rotate_right","orbit"]',
    ),
    (
        "valid_move_instructs = ['up','down','left','right','forward','backward','rotate_left','rotate_right']",
        "valid_move_instructs = ['up','down','left','right','forward','backward','rotate_left','rotate_right','orbit']",
    ),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: apply_flexworld_orbit_traj.py <FLEXWORLD_DIR>", file=sys.stderr)
        return 2
    root = pathlib.Path(sys.argv[1]).resolve()
    path = root / "ops" / "utils" / "all_traj.py"
    if not path.is_file():
        print(f"[ERROR] Not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    force = "--force" in sys.argv or "--refresh" in sys.argv
    if 'move_instruct == "orbit"' in text:
        if force and ORBIT_BRANCH_LEGACY in text:
            text = text.replace(ORBIT_BRANCH_LEGACY, ORBIT_BRANCH, 1)
            path.write_text(text, encoding="utf-8")
            print(f"[OK] Refreshed orbit trajectory (shorter) in {path}")
            return 0
        if force and "num_frames=16)" in text and "num_frames=32)" in text and "orbit" in text:
            text = text.replace(
                ".move_forward(0.1, num_frames=16)",
                ".move_forward(0.1, num_frames=8)",
                1,
            )
            text = text.replace(
                ".move_orbit_to(0, 179.999, 0.1, 0, num_frames=32)",
                ".move_orbit_to(0, 179.999, 0.1, 0, num_frames=16)",
                1,
            )
            path.write_text(text, encoding="utf-8")
            print(f"[OK] Shortened orbit num_frames in {path}")
            return 0
        print(f"[OK] Orbit trajectory already present in {path}")
        return 0

    new_text = text
    for old, new in VALID_REPLACEMENTS:
        if old in new_text:
            new_text = new_text.replace(old, new, 1)
            break
    if new_text == text:
        print("[ERROR] Could not find valid_move_instructs line to extend.", file=sys.stderr)
        return 1

    anchor = 'elif move_instruct == "rotate_right":'
    pos = new_text.find(anchor)
    if pos == -1:
        anchor = "elif move_instruct == 'rotate_right':"
        pos = new_text.find(anchor)
    if pos == -1:
        print("[ERROR] Could not find rotate_right branch.", file=sys.stderr)
        return 1

    sub = new_text[pos:]
    for pat in ("\n    return traj", "\n\treturn traj"):
        rel = sub.find(pat)
        if rel != -1:
            insert_at = pos + rel
            new_text = new_text[:insert_at] + "\n" + ORBIT_BRANCH + new_text[insert_at:]
            path.write_text(new_text, encoding="utf-8")
            print(f"[OK] Patched {path} with orbit trajectory.")
            return 0

    print("[ERROR] Could not find return traj after rotate_right.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
