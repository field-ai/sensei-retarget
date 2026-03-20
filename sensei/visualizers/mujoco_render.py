"""
Render G1 robot motion using MuJoCo offscreen renderer.

Requires: mujoco  (already in the sensei env)
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless EGL — must be set before import

import numpy as np
import mujoco


def render_g1_frames(
    mjcf_path: str,
    root_pos: np.ndarray,    # (N, 3)  xyz
    root_rot: np.ndarray,    # (N, 4)  wxyz quaternion
    dof_pos:  np.ndarray,    # (N, DoF)
    height: int = 480,
    width:  int = 640,
    azimuth:   float = 135.0,   # camera azimuth (deg)
    elevation: float = -12.0,   # camera elevation (deg, negative = looking down)
    distance:  float = 3.5,     # camera distance (m)
    lookat_z_offset: float = 0.75,  # height above root to centre the view
) -> list[np.ndarray]:
    """
    Parameters
    ----------
    mjcf_path : path to the G1 MuJoCo XML
    root_pos  : (N, 3) root position in world frame
    root_rot  : (N, 4) root quaternion wxyz
    dof_pos   : (N, DoF) joint angles
    height, width : render resolution per panel
    azimuth, elevation, distance : camera parameters
    lookat_z_offset : offset above root_pos to place the camera lookat point

    Returns
    -------
    list of (H, W, 3) uint8 RGB frames
    """
    model    = mujoco.MjModel.from_xml_path(mjcf_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    n_dof = dof_pos.shape[1]
    frames: list[np.ndarray] = []

    for i in range(len(dof_pos)):
        # Set full qpos: root_pos (3) + root_quat (4) + joints (DoF)
        data.qpos[0:3]      = root_pos[i]
        data.qpos[3:7]      = root_rot[i]       # wxyz — MuJoCo convention
        data.qpos[7:7+n_dof] = dof_pos[i]

        mujoco.mj_forward(model, data)

        # Camera tracks the robot's root position
        cam = mujoco.MjvCamera()
        cam.lookat[:] = root_pos[i] + np.array([0.0, 0.0, lookat_z_offset])
        cam.distance  = distance
        cam.elevation = elevation
        cam.azimuth   = azimuth

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames
