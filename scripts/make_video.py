#!/usr/bin/env python3
"""
Generate a three-panel visualization video from a GVHMR .pt file.

Panels (left → right):
  1. Original input video  (0_input_video.mp4 from the same directory)
  2. SMPL body mesh        (GVHMR pre-rendered 2_global.mp4 → 1_incam.mp4 → MuJoCo fallback)
  3. Unitree G1 retargeted (MuJoCo offscreen, GMR exact camera: az=180 el=-10 d=2.0)

Layout:
  • Three 640×640 square panels separated by 4 px dark dividers
  • 40 px info bar at the bottom (clip name + frame counter)
  • Letterboxing (not stretching) preserves source aspect ratios
  • Dark charcoal background (#121212) fills letterbox margins

SMPL panel priority:
  1. {clip_dir}/2_global.mp4  — GVHMR global mesh render (best quality)
  2. {clip_dir}/1_incam.mp4   — GVHMR in-camera mesh render
  3. MuJoCo stick figure       — fallback when GVHMR renders are not present

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

# ── Layout constants ──────────────────────────────────────────────────────────
PANEL_SZ  = 640          # each panel is PANEL_SZ × PANEL_SZ (square)
SEP_W     = 4            # width of the divider strip between panels
BAR_H     = 40           # bottom info bar height
_BG       = (18, 18, 18) # letterbox / divider background colour (RGB)

# Label colours (RGB) — one per panel
_LABEL_COLOR = {
    "Input video":   (110, 185, 255),   # cool blue
    "SMPL-X body":   (200, 145, 255),   # lilac
    "G1 retargeted": ( 85, 220, 145),   # mint green
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize preserving aspect ratio; pad to exactly h × w with _BG."""
    src_h, src_w = img.shape[:2]
    scale = min(w / src_w, h / src_h)
    nw = int(src_w * scale)
    nh = int(src_h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), _BG, dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def read_video_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Read up to n_frames, letterbox each to PANEL_SZ × PANEL_SZ."""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = letterbox(frame, PANEL_SZ, PANEL_SZ)
        frames.append(frame)
    cap.release()
    filler = (frames[-1].copy() if frames
              else np.full((PANEL_SZ, PANEL_SZ, 3), _BG, dtype=np.uint8))
    while len(frames) < n_frames:
        frames.append(filler.copy())
    return frames[:n_frames]


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Coloured label with semi-transparent dark background box."""
    out = img.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thick = 2
    pad   = 9
    color = _LABEL_COLOR.get(text, (255, 255, 255))

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x1, y1 = 10, 10
    x2, y2 = x1 + tw + 2 * pad, y1 + th + 2 * pad

    # Semi-transparent black backing
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, out, 0.42, 0, out)

    cv2.putText(out, text, (x1 + pad, y2 - pad // 2),
                font, scale, color, thick, cv2.LINE_AA)
    return out


def make_info_bar(text: str, w: int) -> np.ndarray:
    """A slim dark bar spanning the full output width with centred text."""
    bar = np.full((BAR_H, w, 3), (12, 12, 12), dtype=np.uint8)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.55
    thick  = 1
    color  = (160, 160, 160)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = (BAR_H + th) // 2
    cv2.putText(bar, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
    return bar


def compose_frame(f1: np.ndarray, f2: np.ndarray, f3: np.ndarray,
                  bar_text: str) -> np.ndarray:
    """Stack three panels with separators and attach the info bar."""
    sep = np.full((PANEL_SZ, SEP_W, 3), _BG, dtype=np.uint8)
    row = np.concatenate([f1, sep, f2, sep, f3], axis=1)
    total_w = row.shape[1]
    bar = make_info_bar(bar_text, total_w)
    return np.concatenate([row, bar], axis=0)


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
    clip_name = pt_path.parent.name

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
        orig_frames = [np.full((PANEL_SZ, PANEL_SZ, 3), _BG, dtype=np.uint8)] * N

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
        smpl_frames = render_smpl_frames(
            motion.landmarks, height=PANEL_SZ, width=PANEL_SZ)
        print(f"  {len(smpl_frames)} frames in {time.perf_counter()-t0:.2f}s")

    # ── 5. Panel 3 — G1 robot (MuJoCo, GMR camera) ───────────────────────────
    print("[make_video] rendering G1 robot (MuJoCo, GMR camera) …")
    t0 = time.perf_counter()
    from sensei.visualizers.mujoco_render import render_g1_frames
    g1_frames = render_g1_frames(
        robot.mjcf_path, root_pos, root_rot, dof_pos,
        height=PANEL_SZ, width=PANEL_SZ,
    )
    print(f"  {len(g1_frames)} frames in {time.perf_counter()-t0:.2f}s")

    # ── 6. Compose and write ──────────────────────────────────────────────────
    print(f"[make_video] writing {out_path} …")
    writer = imageio.get_writer(str(out_path), fps=args.fps, macro_block_size=1)
    for idx, (f1, f2, f3) in enumerate(zip(orig_frames, smpl_frames, g1_frames)):
        f1 = add_label(f1, "Input video")
        f2 = add_label(f2, "SMPL-X body")
        f3 = add_label(f3, "G1 retargeted")
        bar_text = f"{clip_name}   frame {idx+1:04d}/{N:04d}   GMR · Unitree G1"
        writer.append_data(compose_frame(f1, f2, f3, bar_text))
    writer.close()

    size_mb = out_path.stat().st_size / 1e6
    total_s = time.perf_counter() - t_total
    print(f"\n[make_video] done  →  {out_path}")
    print(f"  {N} frames @ {args.fps:.0f} fps  |  {size_mb:.1f} MB  |  {total_s:.1f}s total")


if __name__ == "__main__":
    main()
