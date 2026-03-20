"""
GMRSolver: wraps GMR's GeneralMotionRetargeting (mink + DAQP QP solver).

Requires:
    pip install -e third_party/GMR   # installs general_motion_retargeting + mink

Input: MotionSequence with .landmarks populated (use GVHMRSource).
Output: RobotMotion with 29-DoF joint positions for the G1.

qpos layout from GMR.retarget():
    qpos[:3]   = root position (xyz)
    qpos[3:7]  = root quaternion (wxyz)
    qpos[7:]   = 29 joint angles (rad)

We strip root pose and return only the 29 joint angles.
"""
from __future__ import annotations

import numpy as np

from sensei.base.solver import RetargetingSolver
from sensei.types import MotionSequence, RobotConfig, RobotMotion


class GMRSolver(RetargetingSolver):
    """
    Solver backed by GMR's mink differential IK (DAQP QP solver).

    Benchmark baseline: ~60 FPS on a high-end CPU (GMR's published figure).
    Timing is recorded in RobotMotion.metadata['frame_times_s'] by the
    default solve() loop in RetargetingSolver base class.
    """

    def __init__(
        self,
        qp_solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = False,
    ) -> None:
        self._qp_solver = qp_solver
        self._damping = damping
        self._verbose = verbose
        self._gmr = None
        self._robot: RobotConfig | None = None

    @property
    def name(self) -> str:
        return "gmr"

    def setup(self, robot: RobotConfig) -> None:
        # Late import — only fails if GMR not installed
        from general_motion_retargeting import GeneralMotionRetargeting

        self._robot = robot
        gmr_robot_name = robot.metadata.get("gmr_robot_name", robot.name)

        self._gmr = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot=gmr_robot_name,
            actual_human_height=None,   # updated per-sequence in solve()
            solver=self._qp_solver,
            damping=self._damping,
            verbose=self._verbose,
        )

    def solve(self, motion: MotionSequence) -> RobotMotion:
        assert self._gmr is not None, "Call setup() before solve()"
        assert motion.landmarks is not None, (
            "GMRSolver requires MotionSequence.landmarks. Use GVHMRSource."
        )

        # Scale GMR's internal human model to match this sequence's body shape
        human_height = 1.66 + 0.1 * float(motion.betas[0])
        self._update_human_height(human_height)

        # Accumulators for root pose (populated by solve_frame)
        self._root_pos_list: list[np.ndarray] = []
        self._root_rot_list: list[np.ndarray] = []

        result = super().solve(motion)

        # Attach root poses to result metadata
        if self._root_pos_list:
            result.metadata["root_pos"] = np.stack(self._root_pos_list)
            result.metadata["root_rot"] = np.stack(self._root_rot_list)

        return result

    def solve_frame(
        self,
        targets: dict,
        q_prev: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        targets: the per-frame landmark dict from GVHMRSource
            {joint_name: (pos (3,), rot (scipy Rotation))}
        """
        assert self._robot is not None
        try:
            qpos = self._gmr.retarget(targets)          # (7 + DoF,) float64
            dof = self._robot.num_dof
            q = qpos[7 : 7 + dof].astype(np.float64)
            self._root_pos_list.append(qpos[0:3].astype(np.float64))
            self._root_rot_list.append(qpos[3:7].astype(np.float64))
            return q, True
        except Exception:
            prev_pos = self._root_pos_list[-1] if self._root_pos_list else np.zeros(3)
            prev_rot = self._root_rot_list[-1] if self._root_rot_list else np.array([1.,0.,0.,0.])
            self._root_pos_list.append(prev_pos.copy())
            self._root_rot_list.append(prev_rot.copy())
            return q_prev.copy(), False

    def _update_human_height(self, height: float) -> None:
        """
        Update GMR's internal scale table for the given human height.

        GMR bakes height into the scale table at construction time via the
        IK config's human_height_assumption. We scale the table in-place to
        match the current sequence without re-creating the full GMR object.
        """
        if self._gmr is None:
            return
        try:
            # GMR stores the original ratio; re-derive from ik_config assumption
            assumption = self._gmr.ik_config_human_height  # may not exist
            ratio = height / assumption
        except AttributeError:
            # Fall back: re-create GMR with new height (slower, but correct)
            from general_motion_retargeting import GeneralMotionRetargeting
            robot = self._robot
            self._gmr = GeneralMotionRetargeting(
                src_human="smplx",
                tgt_robot=robot.metadata.get("gmr_robot_name", robot.name),
                actual_human_height=height,
                solver=self._qp_solver,
                damping=self._damping,
                verbose=self._verbose,
            )

    def teardown(self) -> None:
        self._gmr = None
        self._robot = None
