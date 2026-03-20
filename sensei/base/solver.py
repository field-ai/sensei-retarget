"""Abstract base class for all retargeting solvers."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from sensei.types import MotionSequence, RobotConfig, RobotMotion


class RetargetingSolver(ABC):
    """
    Retargets a human MotionSequence to robot joint trajectories.

    Lifecycle:
        solver.setup(robot)       — called once; load model, build solver
        result = solver.solve(motion)  — batch solve all frames
        solver.teardown()         — optional cleanup

    solve() has a default implementation that calls solve_frame() in a loop.
    Override solve() for solvers that batch more efficiently.

    solve_frame() is also exposed for streaming/real-time use.

    All solver deps must be imported inside setup() (late import), not at
    module top-level. This keeps the module importable even if the solver's
    dependencies are not installed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in configs, logs, and CLI flags."""
        ...

    @abstractmethod
    def setup(self, robot: RobotConfig) -> None:
        """
        Initialize the solver for the given robot.

        Called exactly once before any solve calls. Load URDF/MJCF, build
        symbolic model, pre-compile cost functions, etc.
        """
        ...

    def solve(self, motion: MotionSequence) -> RobotMotion:
        """
        Batch-solve all frames in `motion`.

        Default implementation: iterate frames, call solve_frame() per frame,
        accumulate RobotState list, record timing in metadata.
        Override for solvers that can vectorise across frames.
        """
        import time

        assert hasattr(self, "_robot") and self._robot is not None, (
            f"{self.__class__.__name__}.setup() must be called before solve()"
        )

        robot: RobotConfig = self._robot
        N = motion.num_frames
        states = []
        converged = np.ones(N, dtype=bool)
        frame_times = np.empty(N, dtype=np.float64)

        q_prev = robot.default_pose.copy()

        for i in range(N):
            targets = self._frame_targets(motion, i)
            t0 = time.perf_counter()
            q, ok = self.solve_frame(targets, q_prev)
            frame_times[i] = time.perf_counter() - t0
            converged[i] = ok

            dq = (q - q_prev) * motion.fps
            states.append(
                __import__("sensei.types", fromlist=["RobotState"]).RobotState(
                    q=q.copy(),
                    dq=dq.copy(),
                    timestamp=i / motion.fps,
                )
            )
            q_prev = q

        return RobotMotion(
            robot_name=robot.name,
            solver_name=self.name,
            fps=motion.fps,
            states=states,
            converged=converged,
            metadata={"frame_times_s": frame_times},
        )

    @abstractmethod
    def solve_frame(
        self,
        targets: dict,
        q_prev: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve a single frame.

        Args:
            targets: solver-specific target dict. For GMR: the landmark dict
                     {joint_name: (pos(3,), rot_mat(3,3))}. For Pinocchio
                     solvers: {ee_label: SE3 (4x4, float64)}.
            q_prev:  (DoF,) previous joint configuration for warm-start.

        Returns:
            q:         (DoF,) solution joint positions, float64.
            converged: True if the solver found a valid solution.
        """
        ...

    def _frame_targets(self, motion: MotionSequence, i: int) -> dict:
        """
        Extract per-frame targets from a MotionSequence.
        Default: return motion.landmarks[i] (GMR format).
        Override for solvers that need a different format.
        """
        if motion.landmarks is not None:
            return motion.landmarks[i]
        raise NotImplementedError(
            f"{self.__class__.__name__} requires MotionSequence.landmarks or "
            "a _frame_targets() override."
        )

    def teardown(self) -> None:
        """Optional cleanup after all solve() calls."""
        pass
