"""
PinocchioIPOPTSolver: full 29-DoF NLP retargeting via Pinocchio + CasADi + IPOPT.

Phase 2a (collision=False): kinematic NLP — EE position + orientation tracking.
Phase 2b (collision=True):  + ground plane + sphere self-collision constraints.

Architecture
------------
Root pose (position + orientation) is extracted from the SMPL-X pelvis target
and fixed as an NLP *parameter* each frame — not optimised over.
The 29 body joints are the optimisation variables.

Implementation note
-------------------
pinocchio.casadi requires SX (symbolic scalars), not MX (symbolic matrices).
We use casadi.nlpsol directly with SX throughout — no casadi.Opti.
All EE targets are stacked into a single parameter vector p_all so that nlpsol
can be called with a plain numpy vector each frame.

Requires
--------
    conda install -c conda-forge pinocchio casadi ipopt
"""
from __future__ import annotations

import numpy as np

from sensei.base.solver import RetargetingSolver
from sensei.types import MotionSequence, RobotConfig, RobotMotion


# ── Target mapping: SMPL-X joint → G1 URDF frame → (pos_weight, rot_weight) ──
# Weights match smplx_to_g1.json ik_match_table2 (GMR's second / refining pass).
# GMR runs two sequential QP passes; we run one NLP so we use the table2 weights
# which represent the final refined cost structure.
#
# Key differences from a naive guess:
#   - Arms (shoulder/elbow/wrist): pos_w=10, NOT 100 — arm shape driven by rotation
#   - Feet: rot_w=50, very strong rotation to nail foot orientation
#   - Spine: pos_w=0, rotation-only
_EE_MAP: list[tuple[str, str, float, float]] = [
    # Feet — strong position + rotation
    ("left_foot",      "left_ankle_roll_link",   100.0, 50.0),
    ("right_foot",     "right_ankle_roll_link",  100.0, 50.0),
    # Wrists
    ("left_wrist",     "left_wrist_yaw_link",     10.0,  5.0),
    ("right_wrist",    "right_wrist_yaw_link",    10.0,  5.0),
    # Elbows
    ("left_elbow",     "left_elbow_link",         10.0,  5.0),
    ("right_elbow",    "right_elbow_link",        10.0,  5.0),
    # Hips
    ("left_hip",       "left_hip_roll_link",      10.0,  5.0),
    ("right_hip",      "right_hip_roll_link",     10.0,  5.0),
    # Knees
    ("left_knee",      "left_knee_link",          10.0,  5.0),
    ("right_knee",     "right_knee_link",         10.0,  5.0),
    # Spine — rotation only
    ("spine3",         "torso_link",               0.0, 10.0),
    # Shoulders
    ("left_shoulder",  "left_shoulder_yaw_link",  10.0,  5.0),
    ("right_shoulder", "right_shoulder_yaw_link", 10.0,  5.0),
]

# 14 finger joints in g1_body29_hand14.urdf — locked to neutral so we get
# a clean 29-DoF body model matching Phase 1 (g1_mocap_29dof.xml)
_HAND_JOINTS: list[str] = [
    "left_hand_index_0_joint",  "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",  "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

_NEE = len(_EE_MAP)   # 13

# ── GMR IK-config preprocessing tables ───────────────────────────────────────
# Derived from smplx_to_g1.json — must match exactly what GMR applies in
# scale_human_data() + offset_human_data() before its own IK.

# Scale factors: applied to each SMPL-X joint position relative to the pelvis
# (pelvis position itself is also scaled by its own factor).
_SCALE: dict[str, float] = {
    "pelvis":         0.9,
    "spine3":         0.9,
    "left_hip":       0.9,  "right_hip":       0.9,
    "left_knee":      0.9,  "right_knee":      0.9,
    "left_foot":      0.9,  "right_foot":      0.9,
    "left_shoulder":  0.8,  "right_shoulder":  0.8,
    "left_elbow":     0.8,  "right_elbow":     0.8,
    "left_wrist":     0.8,  "right_wrist":     0.8,
}

# Rotation offsets (wxyz, scalar-first): right-multiply the raw SMPL-X
# orientation to obtain the target orientation in the robot's frame.
# Identity = [1,0,0,0]; keyed by the SMPL-X joint name used in _EE_MAP.
_ROT_OFFSET_WXYZ: dict[str, list[float]] = {
    "left_foot":      [ 0.5,        -0.5,        -0.5,        -0.5       ],
    "right_foot":     [ 0.5,        -0.5,        -0.5,        -0.5       ],
    "left_wrist":     [ 1.0,         0.0,         0.0,         0.0       ],  # identity
    "right_wrist":    [ 0.0,         0.0,         0.0,        -1.0       ],  # 180° Z
    "left_elbow":     [ 1.0,         0.0,         0.0,         0.0       ],  # identity
    "right_elbow":    [ 0.0,         0.0,         0.0,        -1.0       ],  # 180° Z
    "left_hip":       [ 0.42677550, -0.56379311, -0.56379311, -0.42677550],
    "right_hip":      [ 0.42677550, -0.56379311, -0.56379311, -0.42677550],
    "left_knee":      [ 0.5,        -0.5,        -0.5,        -0.5       ],
    "right_knee":     [ 0.5,        -0.5,        -0.5,        -0.5       ],
    "spine3":         [ 0.5,        -0.5,        -0.5,        -0.5       ],
    "left_shoulder":  [ 0.70710678,  0.0,        -0.70710678,  0.0       ],
    "right_shoulder": [ 0.0,         0.70710678,  0.0,         0.70710678],
}
_PELVIS_ROT_OFFSET_WXYZ: list[float] = [0.5, -0.5, -0.5, -0.5]


class PinocchioIPOPTSolver(RetargetingSolver):
    """
    Full 29-DoF NLP retargeting: Pinocchio FK + CasADi symbolic diff + IPOPT.

    Root pose is taken directly from the SMPL-X pelvis (fixed parameter each
    frame).  The 29 body joints are the optimisation variables.

    Args:
        collision:     enable ground + self-collision constraints (Phase 2b)
        max_iter:      IPOPT max iterations per frame (default 100)
        trans_weight:  global multiplier on position cost terms (default 1.0)
        rot_weight:    global multiplier on rotation cost terms (default 1.0)
        reg_weight:    joint regularisation toward zero (default 0.01)
        smooth_weight: frame-to-frame smoothing weight (default 0.5)
    """

    def __init__(
        self,
        collision:     bool  = False,
        max_iter:      int   = 100,
        trans_weight:  float = 1.0,
        rot_weight:    float = 1.0,
        reg_weight:    float = 0.01,
        smooth_weight: float = 0.5,
    ) -> None:
        self._collision     = collision
        self._max_iter      = max_iter
        self._trans_weight  = trans_weight
        self._rot_weight    = rot_weight
        self._reg_weight    = reg_weight
        self._smooth_weight = smooth_weight

        # set in setup()
        self._nlp_solver    = None
        self._q_lo          = None
        self._q_hi          = None
        self._ndof          = 29
        self._robot_config: RobotConfig | None = None

        # warm-start: IPOPT dual variables from last solve
        self._lam_x0: np.ndarray | None = None
        self._lam_g0: np.ndarray | None = None

        # Accumulated per-frame root poses (for metadata, like GMRSolver)
        self._root_pos_list: list[np.ndarray] = []
        self._root_rot_list: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return "pinocchio_ipopt" + ("_collision" if self._collision else "")

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, robot: RobotConfig) -> None:
        import pathlib
        import pinocchio as pin
        import pinocchio.casadi as cpin
        import casadi

        self._robot_config = robot
        self._robot        = robot   # required by base-class solve() loop
        urdf      = robot.urdf_path
        mesh_dir  = str(pathlib.Path(urdf).parent)

        # 1. Load full URDF → lock 14 hand joints → 29-DoF body model
        robot_full = pin.RobotWrapper.BuildFromURDF(
            urdf, [mesh_dir], pin.JointModelFreeFlyer()
        )
        ref_q    = pin.neutral(robot_full.model)
        robot_29 = robot_full.buildReducedRobot(_HAND_JOINTS, ref_q)
        model    = robot_29.model          # nq=36: 7 (free-flyer) + 29 joints
        ndof     = model.nq - 7            # 29

        # 2. CasADi symbolic model — must use SX throughout
        cmodel = cpin.Model(model)
        cdata  = cmodel.createData()

        # 3. Decision variables and parameters (all SX)
        #
        #    x   = joints (29,)
        #    p   = [root(7), q_last(29), p_tgts(3*NEE), R_tgts(9*NEE)]
        #            0..6     7..35       36..74          75..191
        #
        x_sx  = casadi.SX.sym("q",      ndof)               # 29
        p_sx  = casadi.SX.sym("p",      7 + ndof + 3*_NEE + 9*_NEE)

        root_sx   = p_sx[:7]                                 # (7,)
        qlast_sx  = p_sx[7 : 7 + ndof]                      # (29,)
        ptgt_sx   = p_sx[7 + ndof : 7 + ndof + 3*_NEE]      # (3*NEE,)
        Rtgt_sx   = p_sx[7 + ndof + 3*_NEE :]               # (9*NEE,)

        # 4. FK
        q_full_sx = casadi.vertcat(root_sx, x_sx)            # (36,)
        cpin.framesForwardKinematics(cmodel, cdata, q_full_sx)

        # 5. Cost
        cost_sx = casadi.SX(0)
        for i, (smpl_name, g1_frame, pw, rw) in enumerate(_EE_MAP):
            fid  = cmodel.getFrameId(g1_frame)
            p_fk = cdata.oMf[fid].translation   # SX (3,)
            R_fk = cdata.oMf[fid].rotation       # SX (3,3)

            p_ref = ptgt_sx[3*i : 3*i+3]                    # (3,)
            R_ref = casadi.reshape(Rtgt_sx[9*i : 9*i+9], 3, 3)  # (3,3)

            if pw > 0:
                cost_sx = cost_sx + (self._trans_weight * pw) * casadi.sumsqr(
                    p_fk - p_ref
                )
            if rw > 0:
                err_rot = cpin.log3(R_fk @ R_ref.T)
                cost_sx = cost_sx + (self._rot_weight * rw) * casadi.sumsqr(err_rot)

        cost_sx = cost_sx + self._reg_weight    * casadi.sumsqr(x_sx)
        cost_sx = cost_sx + self._smooth_weight * casadi.sumsqr(x_sx - qlast_sx)

        # 6. Joint limits
        #    Use the tighter of (URDF limits, robot config limits) so that the
        #    NLP respects the operational limits set by the MuJoCo model.
        urdf_lo = np.array(model.lowerPositionLimit[7:], dtype=np.float64)
        urdf_hi = np.array(model.upperPositionLimit[7:], dtype=np.float64)
        _LARGE  = 1e6
        urdf_lo = np.where(np.isinf(urdf_lo), -_LARGE, urdf_lo)
        urdf_hi = np.where(np.isinf(urdf_hi),  _LARGE, urdf_hi)
        # robot config limits come from g1_mocap_29dof.xml — same joint order
        q_lo = np.maximum(robot.joint_lower, urdf_lo)
        q_hi = np.minimum(robot.joint_upper, urdf_hi)

        # 7. Build IPOPT NLP
        nlp  = {"x": x_sx, "p": p_sx, "f": cost_sx}
        opts = {
            "ipopt.print_level":           0,
            "ipopt.max_iter":              self._max_iter,
            "ipopt.tol":                   1e-4,
            "ipopt.acceptable_tol":        5e-4,
            "ipopt.acceptable_iter":       5,
            "ipopt.warm_start_init_point": "yes",
            "print_time":                  0,
        }
        self._nlp_solver = casadi.nlpsol("retarget", "ipopt", nlp, opts)
        self._q_lo       = q_lo
        self._q_hi       = q_hi
        self._ndof       = ndof
        self._p_dim      = int(p_sx.shape[0])

        # Reset warm-start state
        self._lam_x0 = np.zeros(ndof, dtype=np.float64)
        self._lam_g0 = np.zeros(0,    dtype=np.float64)

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self, motion: MotionSequence) -> RobotMotion:
        assert self._nlp_solver is not None, "Call setup() before solve()"
        assert motion.landmarks is not None, (
            "PinocchioIPOPTSolver requires MotionSequence.landmarks. Use GVHMRSource."
        )
        self._root_pos_list = []
        self._root_rot_list = []

        result = super().solve(motion)

        if self._root_pos_list:
            result.metadata["root_pos"] = np.stack(self._root_pos_list)
            result.metadata["root_rot"] = np.stack(self._root_rot_list)
        return result

    def solve_frame(
        self, targets: dict, q_prev: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """
        targets : landmark dict from GVHMRSource
                  {smpl_joint: (pos (3,), rot (scipy.Rotation))}
        q_prev  : (29,) previous joint configuration
        """
        # ── Root from SMPL-X pelvis — with GMR frame corrections ─────────────
        #
        # GMR applies two preprocessing steps to SMPL-X landmarks before IK:
        #   1. scale_human_data : pelvis_pos *= 0.9; other joints scaled relative
        #   2. offset_human_data: orientation right-multiplied by rot_offset (wxyz)
        # We replicate the same transformations so our NLP targets match GMR's.
        from scipy.spatial.transform import Rotation as _R

        if "pelvis" in targets:
            raw_pelvis_pos, raw_pelvis_rot = targets["pelvis"]
            raw_pelvis_pos = raw_pelvis_pos.astype(np.float64)
            # 1. Scale root position
            root_pos = _SCALE["pelvis"] * raw_pelvis_pos
            # 2. Apply rotation offset to pelvis orientation
            rot_p = _R.from_quat(_to_xyzw(raw_pelvis_rot))
            w, x, y, z = _PELVIS_ROT_OFFSET_WXYZ
            rot_p = rot_p * _R.from_quat([x, y, z, w])
            root_quat_xyzw = rot_p.as_quat().astype(np.float64)
        else:
            raw_pelvis_pos = np.zeros(3, dtype=np.float64)
            root_pos       = np.zeros(3, dtype=np.float64)
            root_quat_xyzw = np.array([0., 0., 0., 1.], dtype=np.float64)

        root_vec = np.concatenate([root_pos, root_quat_xyzw]).astype(np.float64)

        # Accumulate for metadata (wxyz for MuJoCo renderer)
        wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0],
                         root_quat_xyzw[1], root_quat_xyzw[2]])
        self._root_pos_list.append(root_pos.copy())
        self._root_rot_list.append(wxyz)

        # ── Build EE parameter vector (positions scaled, orientations offset) ─
        q0   = np.clip(q_prev, -1e6, 1e6)
        ptgt = np.zeros(3 * _NEE, dtype=np.float64)
        Rtgt = np.zeros(9 * _NEE, dtype=np.float64)

        for i, (smpl_name, _, _, _) in enumerate(_EE_MAP):
            if smpl_name not in targets:
                Rtgt[9*i : 9*i+9] = np.eye(3).flatten()
                continue
            raw_pos, raw_rot = targets[smpl_name]
            raw_pos = raw_pos.astype(np.float64)

            # Scale position relative to pelvis (GMR: human_scale_table)
            scale = _SCALE.get(smpl_name, 1.0)
            pos = (raw_pos - raw_pelvis_pos) * scale + root_pos

            # Apply rotation offset (GMR: offset_human_data)
            rot = _R.from_quat(_to_xyzw(raw_rot))
            off = _ROT_OFFSET_WXYZ.get(smpl_name, [1., 0., 0., 0.])
            w, x, y, z = off
            rot = rot * _R.from_quat([x, y, z, w])

            ptgt[3*i : 3*i+3] = pos
            Rtgt[9*i : 9*i+9] = rot.as_matrix().flatten()

        p_all = np.concatenate([root_vec, q0, ptgt, Rtgt])

        # ── Solve ─────────────────────────────────────────────────────────────
        try:
            sol = self._nlp_solver(
                x0   = q0,
                p    = p_all,
                lbx  = self._q_lo,
                ubx  = self._q_hi,
                lam_x0 = self._lam_x0,
                lam_g0 = self._lam_g0,
            )
            q = np.asarray(sol["x"], dtype=np.float64).flatten()
            # Save dual variables for next frame warm-start
            self._lam_x0 = np.asarray(sol["lam_x"], dtype=np.float64).flatten()
            self._lam_g0 = np.asarray(sol["lam_g"], dtype=np.float64).flatten()
            converged = True
        except Exception:
            q = q0.copy()
            converged = False

        return q, converged

    def teardown(self) -> None:
        self._nlp_solver   = None
        self._q_lo         = None
        self._q_hi         = None
        self._robot_config = None
        self._robot        = None
        self._lam_x0       = None
        self._lam_g0       = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_xyzw(rot) -> np.ndarray:
    """Convert rotation to xyzw quaternion (Pinocchio convention).

    Accepts:
      - scipy Rotation
      - (3,3) rotation matrix  → from_matrix
      - (4,) wxyz quaternion   → reorder to xyzw  (GMR landmark format)
      - (4,) xyzw quaternion   → returned as-is   (already Pinocchio format)

    GMR landmarks store quaternions as wxyz (scalar_first=True in scipy).
    We detect this by checking shape only; Pinocchio expects xyzw.
    """
    from scipy.spatial.transform import Rotation
    if isinstance(rot, Rotation):
        return rot.as_quat().astype(np.float64)   # xyzw
    arr = np.asarray(rot, dtype=np.float64)
    if arr.shape == (3, 3):
        return Rotation.from_matrix(arr).as_quat()  # xyzw
    if arr.shape == (4,):
        # GMR stores wxyz — convert to xyzw
        w, x, y, z = arr
        return np.array([x, y, z, w], dtype=np.float64)
    raise ValueError(f"Unrecognised rotation format: shape {arr.shape}")


def _to_matrix(rot) -> np.ndarray:
    """Convert rotation to (3,3) matrix.

    Accepts scipy Rotation, (3,3) array, or (4,) wxyz quaternion (GMR format).
    """
    from scipy.spatial.transform import Rotation
    if isinstance(rot, Rotation):
        return rot.as_matrix().astype(np.float64)
    arr = np.asarray(rot, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (4,):
        # GMR wxyz → scipy xyzw
        w, x, y, z = arr
        return Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float64)
    raise ValueError(f"Unrecognised rotation format: shape {arr.shape}")


# ── Auto-registration ─────────────────────────────────────────────────────────

from sensei.registry import registry  # noqa: E402
registry.register_solver(PinocchioIPOPTSolver)
