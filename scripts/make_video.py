#!/usr/bin/env python3
"""
Generate a three-panel visualization video from a GVHMR .pt file.

Panels (left → right):
  1. Original input video
  2. SMPL-X body skeleton
  3. Unitree G1 retargeted motion (MuJoCo)

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
# Must be set before mujoco (or any package that imports mujoco) is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

PANEL_H = 480
PANEL_W = 640


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_video_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Read up to n_frames from a video file, pad with last frame if shorter."""
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
    # Pad with last frame if video is shorter than motion
    filler = frames[-1].copy() if frames else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    while len(frames) < n_frames:
        frames.append(filler.copy())
    return frames[:n_frames]


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0),    4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video(
    path: pathlib.Path,
    frame_rows: list[np.ndarray],
    fps: float,
) -> None:
    h, w = frame_rows[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frame_rows:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce a three-panel retargeting video.")
    parser.add_argument("--input",  required=True,
                        help="Path to GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True,
                        help="Output video file (.mp4)")
    parser.add_argument("--fps",    type=float, default=30.0)
    parser.add_argument("--azimuth",   type=float, default=135.0,
                        help="G1 camera azimuth in degrees (default 135)")
    parser.add_argument("--elevation", type=float, default=-12.0,
                        help="G1 camera elevation in degrees (default -12)")
    parser.add_argument("--distance",  type=float, default=3.5,
                        help="G1 camera distance in metres (default 3.5)")
    args = parser.parse_args()

    pt_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()

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
    print(f"  {result.num_frames} frames solved in {time.perf_counter()-t0:.2f}s  "
          f"({result.num_frames/(time.perf_counter()-t0):.0f} fps)")

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
        print(f"  read {len(orig_frames)} frames from {video_path.name}")
    else:
        print(f"  [warn] {video_path} not found — using blank panel")
        orig_frames = [np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)] * N

    # ── 4. Panel 2 — SMPL skeleton ────────────────────────────────────────────
    print("[make_video] rendering SMPL skeleton …")
    t0 = time.perf_counter()
    from sensei.visualizers.smpl_skeleton import render_smpl_frames
    smpl_frames = render_smpl_frames(motion.landmarks, height=PANEL_H, width=PANEL_W)
    print(f"  {len(smpl_frames)} frames in {time.perf_counter()-t0:.2f}s")

    # ── 5. Panel 3 — G1 MuJoCo ───────────────────────────────────────────────
    print("[make_video] rendering G1 robot (MuJoCo) …")
    t0 = time.perf_counter()
    from sensei.visualizers.mujoco_render import render_g1_frames
    g1_frames = render_g1_frames(
        robot.mjcf_path,
        root_pos, root_rot, dof_pos,
        height=PANEL_H, width=PANEL_W,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
    )
    print(f"  {len(g1_frames)} frames in {time.perf_counter()-t0:.2f}s")

    # ── 6. Compose and write ──────────────────────────────────────────────────
    print(f"[make_video] composing and writing {out_path} …")
    rows: list[np.ndarray] = []
    for f1, f2, f3 in zip(orig_frames, smpl_frames, g1_frames):
        f1 = add_label(f1, "Input video")
        f2 = add_label(f2, "SMPL-X body")
        f3 = add_label(f3, "G1 retargeted")
        rows.append(np.concatenate([f1, f2, f3], axis=1))

    write_video(out_path, rows, fps=args.fps)

    total_s = time.perf_counter() - total_t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[make_video] done  →  {out_path}")
    print(f"  {N} frames @ {args.fps:.0f} fps  |  {size_mb:.1f} MB  |  {total_s:.1f}s total")


if __name__ == "__main__":
    main()
