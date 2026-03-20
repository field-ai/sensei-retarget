"""
SMPL-X body landmark renderer using MuJoCo offscreen rendering.

Matches the combined GMR + GVHMR visual style:
  - Coloured joint spheres + capsule bones (GMR's coloured-frame scheme)
  - GVHMR-style checkerboard floor (rgb1=.80 .90 .90, rgb2=.60 .70 .70)
  - Same camera setup as GMR's RobotMotionViewer:
      azimuth=180, elevation=-10, lookat=pelvis

Requires: mujoco (already in the sensei env)
MUJOCO_GL=egl must be set before mujoco is first imported.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

# ── Skeleton connectivity (SMPL-X first 22 body joints) ──────────────────────

SMPL_BONES: list[tuple[str, str]] = [
    ("pelvis", "left_hip"),        ("pelvis", "right_hip"),   ("pelvis", "spine1"),
    ("left_hip",  "left_knee"),    ("left_knee",  "left_ankle"),  ("left_ankle",  "left_foot"),
    ("right_hip", "right_knee"),   ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
    ("spine1", "spine2"),          ("spine2",     "spine3"),
    ("spine3", "neck"),            ("neck",       "head"),
    ("spine3", "left_collar"),     ("left_collar",  "left_shoulder"),
    ("left_shoulder",  "left_elbow"),  ("left_elbow",  "left_wrist"),
    ("spine3", "right_collar"),    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]

_LEFT  = frozenset(["left_hip", "left_knee", "left_ankle", "left_foot",
                     "left_collar", "left_shoulder", "left_elbow", "left_wrist"])
_RIGHT = frozenset(["right_hip", "right_knee", "right_ankle", "right_foot",
                     "right_collar", "right_shoulder", "right_elbow", "right_wrist"])

# ── Colours (GMR RGB frame scheme: left=green, right=red, spine=blue-grey) ───
_RGBA_LEFT  = np.array([0.20, 0.80, 0.39, 1.0], dtype=np.float32)
_RGBA_RIGHT = np.array([0.80, 0.20, 0.20, 1.0], dtype=np.float32)
_RGBA_MID   = np.array([0.60, 0.60, 0.80, 1.0], dtype=np.float32)


def _rgba(name: str) -> np.ndarray:
    if name in _LEFT:  return _RGBA_LEFT
    if name in _RIGHT: return _RGBA_RIGHT
    return _RGBA_MID


# ── MuJoCo scene XML — GVHMR checkerboard floor + warm lighting ──────────────
_SCENE_XML = """\
<mujoco model="smpl_vis">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.4 0.6 0.8" rgb2="0.05 0.05 0.1" width="512" height="512"/>
    <texture name="checker" type="2d" builtin="checker"
             rgb1=".80 .90 .90" rgb2=".60 .70 .70" width="512" height="512"/>
    <material name="checker" texture="checker" texrepeat="8 8" reflectance="0.0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="12 12 0.1" material="checker" pos="0 0 0"/>
    <light pos="0 -3 5" dir="0 0.5 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <light pos="3  3 4" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4" specular="0 0 0"/>
  </worldbody>
</mujoco>
"""


def _sphere(scene: mujoco.MjvScene, pos: np.ndarray, r: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, r),
        pos.astype(np.float64),
        np.eye(3).flatten().astype(np.float64),
        rgba,
    )
    scene.ngeom += 1


def _capsule(scene: mujoco.MjvScene, p1: np.ndarray, p2: np.ndarray, r: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r,
                          p1.astype(np.float64), p2.astype(np.float64))
    g.rgba[:] = rgba
    scene.ngeom += 1


def render_smpl_frames(
    landmarks: list[dict],
    height: int = 480,
    width:  int = 640,
    azimuth:   float = 180.0,    # frontal — matches GMR's RobotMotionViewer
    elevation: float = -10.0,    # matches GMR
    distance:  float = 2.5,      # slightly more than GMR's 2.0 to fit full body
    sphere_r:  float = 0.04,
    bone_r:    float = 0.018,
) -> list[np.ndarray]:
    """
    Render SMPL-X landmarks as a coloured stick figure using MuJoCo offscreen
    rendering.  Style matches GMR's RobotMotionViewer + GVHMR's checkerboard floor.

    Parameters
    ----------
    landmarks : list of per-frame dicts {joint_name: (pos (3,), rot)}

    Returns
    -------
    list of (H, W, 3) uint8 RGB frames
    """
    model    = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)

    output: list[np.ndarray] = []
    for frame in landmarks:
        pts: dict[str, np.ndarray] = {name: pos for name, (pos, _) in frame.items()}
        pelvis = pts.get("pelvis", np.zeros(3))

        cam = mujoco.MjvCamera()
        cam.lookat[:] = pelvis        # exact GMR style: cam.lookat = data.xpos[pelvis_id]
        cam.distance  = distance
        cam.elevation = elevation
        cam.azimuth   = azimuth

        renderer.update_scene(data, camera=cam)
        scene = renderer._scene

        for name, pos in pts.items():
            _sphere(scene, pos, sphere_r, _rgba(name))

        for j1, j2 in SMPL_BONES:
            if j1 in pts and j2 in pts:
                _capsule(scene, pts[j1], pts[j2], bone_r, _rgba(j2))

        output.append(renderer.render().copy())

    renderer.close()
    return output
