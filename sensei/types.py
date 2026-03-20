"""
Canonical data types shared across all sources, solvers, and metrics.

Rules:
- All arrays are numpy float64.
- No solver-specific imports anywhere in this file.
- MotionSequence is the contract between sources and solvers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MotionSequence:
    """
    Canonical intermediate representation between any source and any solver.

    body_pose and global_orient use axis-angle (one 3-vector per joint per frame).
    All arrays are float64 numpy.

    landmarks (optional): pre-computed body landmark positions from FK.
      landmarks[i] = {joint_name: (pos (3,), rot_mat (3,3))}
      Populated by sources that run SMPL-X FK (e.g. GVHMRSource).
      Required by GMRSolver; optional for Pinocchio-based solvers.
    """

    fps: float
    num_frames: int                 # N
    num_joints: int                 # J
    body_pose: np.ndarray           # (N, J, 3) axis-angle, float64
    global_orient: np.ndarray       # (N, 3) root orientation axis-angle, float64
    transl: np.ndarray              # (N, 3) root translation in metres, float64
    betas: np.ndarray               # (B,) body shape parameters, float64
    joint_names: list[str]          # length J
    landmarks: list[dict] | None = None  # len N; each {name: (pos(3,), rot(3,3))}
    metadata: dict = field(default_factory=dict)


@dataclass
class RobotConfig:
    """Static description of a robot for use by any solver."""

    name: str                       # e.g. "unitree_g1"
    mjcf_path: str                  # absolute path to MuJoCo .xml (Phase 1)
    urdf_path: str                  # absolute path to URDF (Phase 2+)
    joint_names: list[str]          # DoF names in model order
    joint_lower: np.ndarray         # (DoF,) lower limits, rad, float64
    joint_upper: np.ndarray         # (DoF,) upper limits, rad, float64
    vel_limits: np.ndarray          # (DoF,) velocity limits, rad/s, float64
    end_effectors: dict[str, str]   # label -> link name in the model
    default_pose: np.ndarray        # (DoF,) home/zero joint configuration, float64
    metadata: dict = field(default_factory=dict)

    @property
    def num_dof(self) -> int:
        return len(self.joint_names)


@dataclass
class RobotState:
    """Joint state at a single timestep."""

    q: np.ndarray                   # (DoF,) joint positions, rad, float64
    dq: np.ndarray                  # (DoF,) joint velocities, rad/s, float64
    timestamp: float                # seconds from sequence start


@dataclass
class RobotMotion:
    """Full joint trajectory produced by a solver."""

    robot_name: str
    solver_name: str
    fps: float
    states: list[RobotState]        # length N
    converged: np.ndarray           # (N,) bool — did solver converge each frame?
    metadata: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.states)

    def q_array(self) -> np.ndarray:
        """Return joint positions as (N, DoF) array."""
        return np.array([s.q for s in self.states])

    def dq_array(self) -> np.ndarray:
        """Return joint velocities as (N, DoF) array."""
        return np.array([s.dq for s in self.states])


@dataclass
class MetricResult:
    """Output of a single metric computation."""

    name: str
    value: float                        # scalar summary (e.g. mean FPS)
    unit: str                           # physical unit string, e.g. "fps", "mm", "ms"
    per_frame: np.ndarray | None = None # (N,) frame-level breakdown
    metadata: dict = field(default_factory=dict)  # extra statistics (p95, min, max…)
