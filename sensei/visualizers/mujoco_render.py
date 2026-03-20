"""
Render G1 robot motion using MuJoCo offscreen rendering.

Camera parameters match GMR's RobotMotionViewer exactly:
    distance  = VIEWER_CAM_DISTANCE_DICT['unitree_g1'] = 2.0
    elevation = -10   (slight downward angle)
    azimuth   = 180   (frontal)
    lookat    = data.xpos[model.body('pelvis').id]

Requires: mujoco (already in the sensei env)
MUJOCO_GL=egl must be set before mujoco is first imported.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco


def render_g1_frames(
    mjcf_path: str,
    root_pos:  np.ndarray,    # (N, 3)  xyz
    root_rot:  np.ndarray,    # (N, 4)  wxyz quaternion
    dof_pos:   np.ndarray,    # (N, DoF)
    height: int   = 480,
    width:  int   = 640,
    azimuth:   float = 180.0,   # frontal — GMR's intended value
    elevation: float = -10.0,   # GMR's default
    distance:  float = 2.0,     # VIEWER_CAM_DISTANCE_DICT['unitree_g1']
) -> list[np.ndarray]:
    """
    Render G1 robot motion with GMR's RobotMotionViewer camera style.

    Parameters
    ----------
    mjcf_path : path to the G1 MuJoCo XML
    root_pos  : (N, 3) root position in world frame
    root_rot  : (N, 4) root quaternion wxyz  (MuJoCo convention)
    dof_pos   : (N, DoF) joint angles

    Returns
    -------
    list of (H, W, 3) uint8 RGB frames
    """
    # Inject offscreen framebuffer size before creating the model so that
    # MuJoCo allocates a buffer large enough for height × width renders.
    # We write a patched copy alongside the original so relative mesh paths resolve.
    import xml.etree.ElementTree as ET
    import tempfile
    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    tree = ET.parse(mjcf_path)
    root_el = tree.getroot()
    vis_el = root_el.find("visual")
    if vis_el is None:
        vis_el = ET.SubElement(root_el, "visual")
    glob_el = vis_el.find("global")
    if glob_el is None:
        glob_el = ET.SubElement(vis_el, "global")
    glob_el.set("offheight", str(max(height, int(glob_el.get("offheight", "0")))))
    glob_el.set("offwidth",  str(max(width,  int(glob_el.get("offwidth",  "0")))))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=mjcf_dir, delete=False
    ) as tmp:
        tree.write(tmp.name)
        tmp_path = tmp.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # GMR tracks the pelvis body (ROBOT_BASE_DICT['unitree_g1'] = 'pelvis')
    try:
        pelvis_id = model.body("pelvis").id
    except Exception:
        pelvis_id = 1   # fallback: first non-world body

    n_dof  = dof_pos.shape[1]
    frames: list[np.ndarray] = []

    for i in range(len(dof_pos)):
        data.qpos[0:3]       = root_pos[i]
        data.qpos[3:7]       = root_rot[i]       # wxyz — MuJoCo convention
        data.qpos[7:7+n_dof] = dof_pos[i]

        mujoco.mj_forward(model, data)

        # Exactly mirroring GMR's step():
        #   self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = data.xpos[pelvis_id]
        cam.distance  = distance
        cam.elevation = elevation
        cam.azimuth   = azimuth

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames
