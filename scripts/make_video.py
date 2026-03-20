#!/usr/bin/env python3
"""
Generate a three-panel visualization video from a GVHMR .pt file.

Panels (left → right):
  1. Original input video  (0_input_video.mp4 from the same directory)
  2. SMPL body mesh        (GVHMR pre-rendered 2_global.mp4 → 1_incam.mp4 → MuJoCo fallback)
  3. Unitree G1 retargeted (MuJoCo offscreen, GMR exact camera: az=180 el=-10 d=2.0)

SMPL panel priority:
  1. {clip_dir}/2_global.mp4  — GVHMR global mesh render (best quality)
  2. {clip_dir}/1_incam.mp4   — GVHMR in-camera mesh render
  3. MuJoCo stick figure       — fallback when GVHMR renders are not present

Video output: imageio.get_writer (same as GVHMR)

Usage:
    python scripts/make_video.py \\
        --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \\
        --output outputs/tennis_vis.mp4

    # All four clips
    for clip in tennis basketball_clip dance_clip 0_input_video; do
        python scripts/make_video.py \\
            --input /mnt/code/GVHMR/outputs/demo/${clip}/hmr4d_results.pt \\
            --output outputs/${clip}_vis.mp4
    done
"""
from __future__ import annotations

import os
# Must be set before mujoco (or any package that imports mujoco) is first imported
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pathlib
import sys
import time

import cv2
import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

PANEL_H = 480
PANEL_W = 640


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_video_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Read up to n_frames from a video, pad with last frame if shorter."""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (PANEL_W, PANEL_H))
        frames.append(frame)
    cap.release()
    filler = frames[-1].copy() if frames else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    while len(frames) < n_frames:
        frames.append(filler.copy())
    return frames[:n_frames]


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Stamp a white label with black outline in the top-left corner."""
    out = img.copy()
    cv2.putText(out, text, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0),    4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-panel retargeting video: input | SMPL body | G1 robot")
    parser.add_argument("--input",  required=True,  help="Path to GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True,  help="Output video (.mp4)")
    parser.add_argument("--fps",    type=float, default=30.0)
    args = parser.parse_args()

    pt_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # ── 1. Load SMPL motion ───────────────────────────────────────────────────
    print("[make_video] loading SMPL motion …")
    from sensei.sources.gvhmr import GVHMRSource
    motion = GVHMRSource().load(str(pt_path))
    N = motion.num_frames
    print(f"  {N} frames @ {motion.fps:.0f} fps")

    # ── 2. Solve IK → G1 poses ────────────────────────────────────────────────
    print("[make_video] solving IK (GMR → G1) …")
    from sensei.solvers.gmr import GMRSolver
    from sensei.robots.g1 import get_g1_config

    robot  = get_g1_config()
    solver = GMRSolver(verbose=False)
    solver.setup(robot)
    t0 = time.perf_counter()
    result = solver.solve(motion)
    solver.teardown()
    elapsed = time.perf_counter() - t0
    print(f"  {result.num_frames} frames in {elapsed:.2f}s  ({result.num_frames/elapsed:.0f} fps)")

    root_pos = result.metadata.get("root_pos", np.zeros((N, 3), dtype=np.float64))
    root_rot = result.metadata.get(
        "root_rot",
        np.tile([1., 0., 0., 0.], (N, 1)).astype(np.float64),
    )
    dof_pos = result.q_array()

    # ── 3. Panel 1 — original video ───────────────────────────────────────────
    print("[make_video] reading original video …")
    video_path = pt_path.parent / "0_input_video.mp4"
    if video_path.exists():
        orig_frames = read_video_frames(str(video_path), N)
        print(f"  {len(orig_frames)} frames from {video_path.name}")
    else:
        print(f"  [warn] {video_path} not found — blank panel")
        orig_frames = [np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)] * N

    # ── 4. Panel 2 — SMPL body (GVHMR pre-render preferred, MuJoCo fallback) ──
    smpl_frames: list[np.ndarray] = []
    for candidate in ("2_global.mp4", "1_incam.mp4"):
        candidate_path = pt_path.parent / candidate
        if candidate_path.exists():
            print(f"[make_video] SMPL panel: reading {candidate_path.name} (GVHMR render) …")
            smpl_frames = read_video_frames(str(candidate_path), N)
            print(f"  {len(smpl_frames)} frames")
            break

    if not smpl_frames:
        print("[make_video] SMPL panel: no GVHMR render found — using MuJoCo stick figure …")
        t0 = time.perf_counter()
        from sensei.visualizers.smpl_mujoco import render_smpl_frames
        smpl_frames = render_smpl_frames(motion.landmarks, height=PANEL_H, width=PANEL_W)
        print(f"  {len(smpl_frames)} frames in {time.perf_counter()-t0:.2f}s")

    # ── 5. Panel 3 — G1 robot (MuJoCo, GMR camera) ───────────────────────────
    print("[make_video] rendering G1 robot (MuJoCo, GMR camera) …")
    t0 = time.perf_counter()
    from sensei.visualizers.mujoco_render import render_g1_frames
    g1_frames = render_g1_frames(
        robot.mjcf_path, root_pos, root_rot, dof_pos,
        height=PANEL_H, width=PANEL_W,
    )
    print(f"  {len(g1_frames)} frames in {time.perf_counter()-t0:.2f}s")

    # ── 6. Compose and write (imageio — same as GVHMR) ───────────────────────
    print(f"[make_video] writing {out_path} …")
    writer = imageio.get_writer(str(out_path), fps=args.fps, macro_block_size=1)
    for f1, f2, f3 in zip(orig_frames, smpl_frames, g1_frames):
        f1 = add_label(f1, "Input video")
        f2 = add_label(f2, "SMPL-X body")
        f3 = add_label(f3, "G1 retargeted")
        writer.append_data(np.concatenate([f1, f2, f3], axis=1))
    writer.close()

    size_mb = out_path.stat().st_size / 1e6
    total_s = time.perf_counter() - t_total
    print(f"\n[make_video] done  →  {out_path}")
    print(f"  {N} frames @ {args.fps:.0f} fps  |  {size_mb:.1f} MB  |  {total_s:.1f}s total")


if __name__ == "__main__":
    main()
