"""
Render SMPL-X body landmarks as an annotated skeleton video using cv2.

Requires: opencv-python  (already in the sensei env)
"""
from __future__ import annotations

import numpy as np
import cv2

# SMPL-X body joint connectivity (parent, child) — first 22 joints
SMPL_BONES = [
    ("pelvis", "left_hip"),        ("pelvis", "right_hip"),   ("pelvis", "spine1"),
    ("left_hip",    "left_knee"),  ("left_knee",  "left_ankle"),  ("left_ankle",  "left_foot"),
    ("right_hip",   "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
    ("spine1",  "spine2"),         ("spine2",     "spine3"),
    ("spine3",  "neck"),           ("neck",       "head"),
    ("spine3",  "left_collar"),    ("left_collar",  "left_shoulder"),
    ("left_shoulder",  "left_elbow"),  ("left_elbow",  "left_wrist"),
    ("spine3",  "right_collar"),   ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]

_LEFT  = {"left_hip", "left_knee", "left_ankle", "left_foot",
           "left_collar", "left_shoulder", "left_elbow", "left_wrist"}
_RIGHT = {"right_hip", "right_knee", "right_ankle", "right_foot",
           "right_collar", "right_shoulder", "right_elbow", "right_wrist"}
_SPINE = {"pelvis", "spine1", "spine2", "spine3", "neck", "head"}

_BONE_COLOR = {
    "left":  (80,  200, 120),   # green
    "right": (200,  80,  80),   # red
    "mid":   (200, 200,  80),   # yellow
}
_JOINT_COLOR = {
    "left":  (100, 220, 150),
    "right": (220, 100, 100),
    "mid":   (220, 220, 100),
}


def _side(name: str) -> str:
    if name in _LEFT:  return "left"
    if name in _RIGHT: return "right"
    return "mid"


def _build_view_matrix() -> np.ndarray:
    """
    Slight 3/4 front view: rotate 20° around the vertical axis so the
    skeleton reads well even for side-on motions.
    """
    theta = np.radians(20)
    c, s = np.cos(theta), np.sin(theta)
    # Rotate around Y (vertical)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Ry


_VIEW = _build_view_matrix()


def render_smpl_frames(
    landmarks: list[dict],
    height: int = 480,
    width: int = 640,
    bg_color: tuple[int, int, int] = (20, 20, 30),
) -> list[np.ndarray]:
    """
    Parameters
    ----------
    landmarks : list of per-frame dicts  {joint_name: (pos (3,), rot)}
    height, width : output panel size
    bg_color : RGB background

    Returns
    -------
    list of (H, W, 3) uint8 RGB frames
    """
    # ── Gather all joint positions to compute stable bounding box ─────────────
    all_pts: list[np.ndarray] = []
    for frame in landmarks:
        for _, (pos, _) in frame.items():
            all_pts.append(_VIEW @ pos)   # rotated 3-D point

    all_pts_arr = np.array(all_pts)       # (M, 3)
    # Project: use X (horizontal) and Y (vertical, flip for image)
    proj_x = all_pts_arr[:, 0]
    proj_y = all_pts_arr[:, 1]

    xmin, xmax = proj_x.min(), proj_x.max()
    ymin, ymax = proj_y.min(), proj_y.max()
    pad_frac = 0.18
    x_span = (xmax - xmin) * (1 + 2 * pad_frac) or 1.0
    y_span = (ymax - ymin) * (1 + 2 * pad_frac) or 1.0
    span = max(x_span, y_span)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    scale = min(height, width) / span

    def to_px(pt3d: np.ndarray) -> tuple[int, int]:
        rv = _VIEW @ pt3d
        px = int((rv[0] - cx) * scale + width  / 2)
        py = int(height / 2 - (rv[1] - cy) * scale)   # flip Y
        return (px, py)

    # ── Render per frame ──────────────────────────────────────────────────────
    output: list[np.ndarray] = []
    for frame in landmarks:
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)

        pts2d: dict[str, tuple[int, int]] = {
            name: to_px(pos)
            for name, (pos, _) in frame.items()
        }

        # Draw bones
        for j1, j2 in SMPL_BONES:
            if j1 in pts2d and j2 in pts2d:
                side = _side(j2)
                cv2.line(img, pts2d[j1], pts2d[j2], _BONE_COLOR[side], 2, cv2.LINE_AA)

        # Draw joints
        for name, px in pts2d.items():
            cv2.circle(img, px, 4, _JOINT_COLOR[_side(name)], -1, cv2.LINE_AA)

        output.append(img)

    return output
