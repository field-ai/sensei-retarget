# Phase 1 — GVHMR + GMR Baseline

> **Goal**: Run the existing GVHMR → GMR pipeline through the new `sensei` abstractions and collect timing metrics. By the end of Phase 1, `scripts/run_pipeline.py` produces a robot motion `.pkl` and a timing report on the four pre-processed test clips.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [What Already Exists](#what-already-exists)
3. [Data Flow](#data-flow)
4. [Phase 0 Scaffold (must come first)](#phase-0-scaffold)
5. [Implementation: GVHMRSource](#implementation-gvhmrsource)
6. [Implementation: GMRSolver](#implementation-gmrsolver)
7. [Implementation: SolverTimingMetric](#implementation-solvertimingmetric)
8. [Implementation: G1 RobotConfig](#implementation-g1-robotconfig)
9. [Wire It: pipeline.py and run_pipeline.py](#wire-it)
10. [Tests](#tests)
11. [Acceptance Criteria](#acceptance-criteria)
12. [Common Pitfalls](#common-pitfalls)

---

## Prerequisites

```bash
# Install GMR (installs mink, mujoco, smplx, qpsolvers[daqp] as transitive deps)
pip install -e /mnt/code/GMR

# Install sensei itself (editable)
pip install -e /mnt/code/sensei-retarget

# Verify
python -c "from general_motion_retargeting import GeneralMotionRetargeting; print('GMR ok')"
python -c "import mink; print('mink ok')"
```

SMPL-X body model files must exist at:
```
/mnt/code/GMR/assets/body_models/smplx/
```

---

## What Already Exists

### Pre-processed GVHMR outputs (ready to use)

```
/mnt/code/GVHMR/outputs/demo/
├── tennis/hmr4d_results.pt
├── basketball_clip/hmr4d_results.pt
├── dance_clip/hmr4d_results.pt
└── 0_input_video/hmr4d_results.pt
```

These are `torch.save()` outputs. Structure:

```python
data = torch.load("hmr4d_results.pt")
# data["smpl_params_global"] is a dict:
#   "body_pose"     (N, 63)   — 21 joints × 3 axis-angle (NOT root), float32 tensor
#   "betas"         (1, 10)   — body shape, float32 tensor
#   "global_orient" (N,  3)   — root axis-angle, float32 tensor
#   "transl"        (N,  3)   — root translation in metres, float32 tensor
# N varies by clip (~100–600 frames at 30 fps)
```

### GMR library (`/mnt/code/GMR`)

Key classes and functions we will wrap:

```python
# Load GVHMR file + run SMPL-X FK → smplx_data, body_model, smplx_output, human_height
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

# Main IK solver
from general_motion_retargeting import GeneralMotionRetargeting

# Construct:
retarget = GeneralMotionRetargeting(
    src_human="smplx",          # IK config key — GVHMR outputs go through SMPL-X model
    tgt_robot="unitree_g1",     # Uses /mnt/code/GMR/assets/unitree_g1/g1_mocap_29dof.xml
    actual_human_height=1.72,   # from GVHMR betas: 1.66 + 0.1 * betas[0]
    solver="daqp",
    damping=5e-1,
    verbose=False,
)

# Per-frame solve:
qpos = retarget.retarget(smplx_data_frame)
# qpos shape: (nq,) where nq = 7 + 29 = 36
# qpos[:3]  = root position (xyz)
# qpos[3:7] = root quaternion (wxyz)
# qpos[7:]  = 29 joint angles (rad)
```

GMR's SMPL-X body model path:
```python
SMPLX_FOLDER = pathlib.Path("/mnt/code/GMR/assets/body_models")
```

---

## Data Flow

```
[.pt file]
    │
    ▼ GVHMRSource.load()
    │  • torch.load() → smpl_params_global
    │  • smplx.create() → SMPL-X body model
    │  • body_model(pose, shape, transl) → smplx_output  [FK forward pass]
    │  • get_gvhmr_data_offline_fast() → list of per-frame landmark dicts
    │    Each dict: {joint_name: (pos: (3,) float64, rot_mat: (3,3) float64)}
    │
    ▼ MotionSequence
    │  .fps, .num_frames
    │  .body_pose  (N, 21, 3) float64  ← raw axis-angle from SMPL params
    │  .global_orient (N, 3) float64
    │  .transl (N, 3) float64
    │  .betas (10,) float64
    │  .landmarks: list[dict]  ← pre-computed from SMPL-X FK (required for GMRSolver)
    │
    ▼ GMRSolver.solve(motion)
    │  • setup() creates GeneralMotionRetargeting("smplx", "unitree_g1")
    │  • for each frame: retarget(motion.landmarks[i]) → qpos (36,)
    │  • times each frame
    │
    ▼ RobotMotion
    │  .states: list[RobotState]  ← q (29,), dq (29,) float64
    │  .converged: (N,) bool
    │
    ▼ SolverTimingMetric.compute()
       • fps, latency_p50, latency_p95, latency_p99 (ms)
```

---

## Phase 0 Scaffold

Before writing Phase 1 code, create the package skeleton. The exact files are listed in `plan.md`. Minimal set required for Phase 1:

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools >= 61"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "sensei-retarget"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy >= 1.24",
    "scipy >= 1.10",
]

[project.optional-dependencies]
gmr = [
    "mink",
    "mujoco",
    "qpsolvers[daqp,proxqp]",
    "smplx",
    "torch",           # needed only for loading GVHMR .pt files
]
dev = [
    "pytest",
    "ruff",
]
```

### `sensei/types.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MotionSequence:
    """
    Canonical intermediate representation between any source and any solver.
    All arrays are float64 numpy. Rotations are axis-angle (N, J, 3).
    landmarks is a pre-computed list of per-frame body landmark dicts — computed by
    the source if it can, consumed by solvers that need Cartesian targets.
    """
    fps: float
    num_frames: int                        # N
    num_joints: int                        # J
    body_pose: np.ndarray                  # (N, J, 3) axis-angle, float64
    global_orient: np.ndarray             # (N, 3) root orientation, float64
    transl: np.ndarray                    # (N, 3) root translation metres, float64
    betas: np.ndarray                     # (B,) shape params, float64
    joint_names: list[str]                # length J
    # pre-computed body landmarks: landmarks[i][joint_name] = (pos (3,), rot_mat (3,3))
    landmarks: list[dict] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RobotConfig:
    name: str
    urdf_path: str                        # absolute path
    mjcf_path: str                        # absolute path to MuJoCo .xml
    joint_names: list[str]                # DoF names in order
    joint_lower: np.ndarray              # (DoF,) float64, rad
    joint_upper: np.ndarray              # (DoF,) float64, rad
    vel_limits: np.ndarray               # (DoF,) float64, rad/s
    end_effectors: dict[str, str]        # label → link name
    default_pose: np.ndarray             # (DoF,) float64, home position


@dataclass
class RobotState:
    q: np.ndarray                        # (DoF,) joint positions, float64
    dq: np.ndarray                       # (DoF,) joint velocities, float64
    timestamp: float                     # seconds from start of sequence


@dataclass
class RobotMotion:
    robot_name: str
    solver_name: str
    fps: float
    states: list[RobotState]
    converged: np.ndarray               # (N,) bool
    metadata: dict = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    value: float
    unit: str
    per_frame: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
```

### `sensei/base/source.py`

```python
from abc import ABC, abstractmethod
from sensei.types import MotionSequence

class MotionSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, path: str) -> MotionSequence: ...

    @abstractmethod
    def can_load(self, path: str) -> bool: ...
```

### `sensei/base/solver.py`

```python
from abc import ABC, abstractmethod
import numpy as np
from sensei.types import MotionSequence, RobotConfig, RobotMotion

class RetargetingSolver(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def setup(self, robot: RobotConfig) -> None: ...

    @abstractmethod
    def solve(self, motion: MotionSequence) -> RobotMotion: ...

    @abstractmethod
    def solve_frame(
        self,
        targets: dict[str, np.ndarray],
        q_prev: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        targets: {label: SE3 (4x4 float64)} end-effector pose targets
        q_prev:  (DoF,) previous solution for warm-start
        Returns: (q, converged)
        """
        ...

    def teardown(self) -> None:
        pass
```

### `sensei/base/metric.py`

```python
from abc import ABC, abstractmethod
from sensei.types import MotionSequence, RobotMotion, MetricResult

class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def unit(self) -> str: ...

    @abstractmethod
    def compute(self, source: MotionSequence, result: RobotMotion) -> MetricResult: ...
```

---

## Implementation: GVHMRSource

**File**: `sensei/sources/gvhmr.py`

```python
"""
GVHMRSource: loads GVHMR hmr4d_results.pt files and converts to MotionSequence.

Delegates SMPL-X FK to GMR's utility functions (avoid re-implementing).
Requires GMR installed: pip install -e /mnt/code/GMR
"""
from __future__ import annotations
import pathlib
import numpy as np
from sensei.base.source import MotionSource
from sensei.types import MotionSequence

# GMR ships SMPL-X body model files here:
_DEFAULT_SMPLX_FOLDER = pathlib.Path("/mnt/code/GMR/assets/body_models")
_TGT_FPS = 30


class GVHMRSource(MotionSource):
    """
    Loads a GVHMR hmr4d_results.pt file.

    Internally calls GMR's load_gvhmr_pred_file + get_gvhmr_data_offline_fast
    to run SMPL-X FK and produce body landmark dicts that GMRSolver consumes.

    Args:
        smplx_folder: path to directory containing the smplx body model.
                      Defaults to /mnt/code/GMR/assets/body_models.
        target_fps:   downsample to this frame rate (default 30).
    """

    def __init__(
        self,
        smplx_folder: str | pathlib.Path = _DEFAULT_SMPLX_FOLDER,
        target_fps: int = _TGT_FPS,
    ) -> None:
        self._smplx_folder = pathlib.Path(smplx_folder)
        self._target_fps = target_fps

    @property
    def name(self) -> str:
        return "gvhmr"

    def can_load(self, path: str) -> bool:
        p = pathlib.Path(path)
        return p.suffix == ".pt" and p.exists()

    def load(self, path: str) -> MotionSequence:
        # Late imports — only fail if GMR or smplx not installed
        from general_motion_retargeting.utils.smpl import (
            load_gvhmr_pred_file,
            get_gvhmr_data_offline_fast,
        )

        smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
            path, self._smplx_folder
        )

        frames_list, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=self._target_fps
        )
        # frames_list: list of dicts {joint_name: (pos (3,), rot_mat (3,3))}
        # All arrays are already float64 numpy after GMR's processing.

        N = len(frames_list)
        # Extract raw SMPL params (float32 → float64)
        body_pose = smplx_data["pose_body"].astype(np.float64)       # (N_orig, 63)
        global_orient = smplx_data["root_orient"].astype(np.float64) # (N_orig, 3)
        transl = smplx_data["trans"].astype(np.float64)               # (N_orig, 3)
        betas = smplx_data["betas"].astype(np.float64).flatten()[:10] # (10,)

        # body_pose is (N_orig, 63) = (N_orig, 21, 3) — reshape
        # Note: N_orig may differ from N (after FPS alignment); use N from frames_list
        # For the raw params we store the aligned count from landmarks
        body_pose_reshaped = body_pose.reshape(-1, 21, 3)[:N]   # (N, 21, 3)
        global_orient = global_orient[:N]
        transl = transl[:N]

        # Canonical joint names from GMR
        from general_motion_retargeting.utils.smpl import JOINT_NAMES
        joint_names = list(JOINT_NAMES[:21])

        return MotionSequence(
            fps=float(aligned_fps),
            num_frames=N,
            num_joints=21,
            body_pose=body_pose_reshaped,
            global_orient=global_orient,
            transl=transl,
            betas=betas,
            joint_names=joint_names,
            landmarks=frames_list,           # list[dict] — consumed by GMRSolver
            metadata={
                "source_path": str(path),
                "human_height": float(human_height),
                "smplx_folder": str(self._smplx_folder),
            },
        )
```

**Notes**:
- `get_gvhmr_data_offline_fast` returns a list of dicts. Each dict maps GMR body landmark names (e.g. `"Hips"`, `"LeftWrist"`) to `(pos, rot_mat)` tuples where `pos` is `(3,)` float64 and `rot_mat` is a `scipy.spatial.transform.Rotation` (or rotation matrix — check the actual return type in GMR source).
- The FPS alignment (SLERP interpolation) is handled entirely inside `get_gvhmr_data_offline_fast`.
- No CUDA required for this path — `torch.load()` and SMPL-X FK run on CPU.

---

## Implementation: GMRSolver

**File**: `sensei/solvers/gmr.py`

```python
"""
GMRSolver: wraps GMR's GeneralMotionRetargeting (mink differential IK).

Requires: pip install -e /mnt/code/GMR
"""
from __future__ import annotations
import time
import numpy as np
from sensei.base.solver import RetargetingSolver
from sensei.types import MotionSequence, RobotConfig, RobotMotion, RobotState


class GMRSolver(RetargetingSolver):
    """
    Solver backed by GMR's GeneralMotionRetargeting (mink + DAQP QP solver).

    Per-frame: calls retarget(landmark_dict) → qpos (36,)
    qpos layout: [root_pos(3), root_quat_wxyz(4), joint_angles(29)]

    We strip root pose (qpos[:7]) and return only the 29 joint angles as q.
    """

    def __init__(
        self,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = False,
    ) -> None:
        self._solver_name_qp = solver
        self._damping = damping
        self._verbose = verbose
        self._gmr = None
        self._robot: RobotConfig | None = None

    @property
    def name(self) -> str:
        return "gmr"

    def setup(self, robot: RobotConfig) -> None:
        # Late import
        from general_motion_retargeting import GeneralMotionRetargeting

        self._robot = robot
        # GMR uses its own robot name — map from RobotConfig.name
        # "unitree_g1" is the GMR key for the G1 29-DoF mocap model
        gmr_robot_name = robot.metadata.get("gmr_robot_name", robot.name)

        self._gmr = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot=gmr_robot_name,
            actual_human_height=None,  # set per-sequence in solve()
            solver=self._solver_name_qp,
            damping=self._damping,
            verbose=self._verbose,
        )

    def solve(self, motion: MotionSequence) -> RobotMotion:
        assert self._gmr is not None, "Call setup() before solve()"
        assert motion.landmarks is not None, (
            "GMRSolver requires MotionSequence.landmarks (body landmark dicts). "
            "Use GVHMRSource or another source that computes landmarks."
        )

        # Update human height scaling from this sequence's betas
        human_height = 1.66 + 0.1 * float(motion.betas[0])
        self._gmr.set_human_height(human_height)  # if GMR supports this
        # Note: if GMR doesn't expose set_human_height(), re-create _gmr with height.
        # Check GeneralMotionRetargeting.__init__ signature.

        N = motion.num_frames
        dof = len(self._robot.joint_names)
        states: list[RobotState] = []
        converged = np.ones(N, dtype=bool)
        frame_times: list[float] = []

        q_prev = self._robot.default_pose.copy()

        for i, landmark_dict in enumerate(motion.landmarks):
            t0 = time.perf_counter()
            try:
                qpos = self._gmr.retarget(landmark_dict)   # (nq,) float64
                dt = time.perf_counter() - t0

                # qpos = [root_pos(3), root_quat_wxyz(4), joint_angles(dof)]
                q = qpos[7:7 + dof].astype(np.float64)
                dq = (q - q_prev) * motion.fps  # finite-difference velocity

                states.append(RobotState(q=q, dq=dq, timestamp=i / motion.fps))
                q_prev = q
            except Exception as e:
                # Solver failed this frame — use previous q
                dt = time.perf_counter() - t0
                converged[i] = False
                q = q_prev.copy()
                states.append(RobotState(q=q, dq=np.zeros(dof), timestamp=i / motion.fps))

            frame_times.append((time.perf_counter() - t0 + dt) / 2)  # rough estimate
            # Note: timing is measured more accurately in SolverTimingMetric

        return RobotMotion(
            robot_name=self._robot.name,
            solver_name=self.name,
            fps=motion.fps,
            states=states,
            converged=converged,
            metadata={"frame_times_s": np.array(frame_times)},
        )

    def solve_frame(
        self,
        targets: dict[str, np.ndarray],
        q_prev: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        targets here should be the GMR-format landmark dict:
          {body_name: (pos (3,), rot_mat (3,3))}
        """
        assert self._gmr is not None
        try:
            qpos = self._gmr.retarget(targets)
            dof = len(self._robot.joint_names)
            return qpos[7:7 + dof].astype(np.float64), True
        except Exception:
            return q_prev.copy(), False

    def teardown(self) -> None:
        self._gmr = None
```

**Important detail**: `GMR.retarget()` takes the landmark dict directly (the same format returned by `get_gvhmr_data_offline_fast` per frame). Do not re-wrap. Verify by checking:

```python
# In GMR motion_retarget.py line ~173:
def retarget(self, human_data, offset_to_ground=False):
    self.update_targets(human_data, offset_to_ground)
    ...
```

**Human height**: `GeneralMotionRetargeting.__init__` takes `actual_human_height` at construction time. If we need to update it per-sequence without re-creating, we need to call the internal scale update directly. Simplest solution: re-create `self._gmr` in `solve()` if height changed. Check GMR source for `human_scale_table`.

---

## Implementation: SolverTimingMetric

**File**: `sensei/metrics/timing.py`

```python
"""
SolverTimingMetric: measures solver wall-clock FPS and per-frame latency.

The metric re-runs the solver with time.perf_counter() wrapping each solve_frame()
call. It reads frame_times from RobotMotion.metadata if already present, otherwise
recomputes.
"""
from __future__ import annotations
import numpy as np
from sensei.base.metric import Metric
from sensei.types import MotionSequence, RobotMotion, MetricResult


class SolverTimingMetric(Metric):
    @property
    def name(self) -> str:
        return "solver_timing"

    @property
    def unit(self) -> str:
        return "ms"

    def compute(self, source: MotionSequence, result: RobotMotion) -> MetricResult:
        frame_times = result.metadata.get("frame_times_s")
        if frame_times is None:
            raise ValueError(
                "SolverTimingMetric requires RobotMotion.metadata['frame_times_s']. "
                "Ensure the solver records per-frame timing."
            )

        frame_times_ms = np.array(frame_times) * 1000.0  # convert s → ms
        fps = 1000.0 / float(np.mean(frame_times_ms))

        return MetricResult(
            name=self.name,
            value=fps,
            unit="fps",
            per_frame=frame_times_ms,
            metadata={
                "latency_p50_ms": float(np.percentile(frame_times_ms, 50)),
                "latency_p95_ms": float(np.percentile(frame_times_ms, 95)),
                "latency_p99_ms": float(np.percentile(frame_times_ms, 99)),
                "latency_mean_ms": float(np.mean(frame_times_ms)),
                "latency_min_ms": float(np.min(frame_times_ms)),
                "latency_max_ms": float(np.max(frame_times_ms)),
                "fps": fps,
                "num_frames": len(frame_times_ms),
                "converge_rate": float(np.mean(result.converged)),
            },
        )
```

**Improve timing accuracy**: For a cleaner measurement, add a separate `TimedSolverWrapper` class that instruments any solver:

```python
# sensei/metrics/timing.py (additional class)
import time

class TimedSolverWrapper(RetargetingSolver):
    """Wraps any solver and records per-frame wall-clock time."""

    def __init__(self, solver: RetargetingSolver) -> None:
        self._inner = solver
        self._frame_times: list[float] = []

    @property
    def name(self) -> str:
        return self._inner.name

    def setup(self, robot) -> None:
        self._inner.setup(robot)

    def solve(self, motion: MotionSequence) -> RobotMotion:
        self._frame_times = []
        result = self._inner.solve(motion)
        result.metadata["frame_times_s"] = np.array(
            result.metadata.get("frame_times_s", self._frame_times)
        )
        return result

    def solve_frame(self, targets, q_prev):
        t0 = time.perf_counter()
        q, converged = self._inner.solve_frame(targets, q_prev)
        self._frame_times.append(time.perf_counter() - t0)
        return q, converged

    def teardown(self) -> None:
        self._inner.teardown()
```

---

## Implementation: G1 RobotConfig

**File**: `sensei/robots/g1.py`

```python
"""
G1 RobotConfig for sensei.

Uses GMR's bundled MuJoCo XML for Phase 1 (mink IK).
URDF path for Phase 2+ (Pinocchio) must be set separately.

Joint names and limits from:
  /mnt/code/GMR/assets/unitree_g1/g1_mocap_29dof.xml
  /mnt/code/sensei-retarget/reference_repos/xr_teleoperate/teleop/robot_control/robot_arm_ik.py
"""
import pathlib
import numpy as np
from sensei.types import RobotConfig

_GMR_ROOT = pathlib.Path("/mnt/code/GMR")

# G1 29-DoF joint names (from GMR MuJoCo model)
G1_JOINT_NAMES = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Approximate joint limits (rad) — verify against URDF
# Source: Unitree G1 technical documentation + G1Pilot / xr_teleoperate
G1_JOINT_LOWER = np.array([
    -2.87, -0.52, -2.09, -0.26, -0.87, -0.35,   # left leg
    -2.87, -2.05, -2.09, -0.26, -0.87, -0.35,   # right leg
    -2.09, -0.26, -0.52,                          # waist
    -2.87, -1.57, -1.57, -1.57, -1.57, -0.52, -1.57,  # left arm
    -2.87, -2.35, -1.57, -1.57, -1.57, -0.52, -1.57,  # right arm
], dtype=np.float64)

G1_JOINT_UPPER = np.array([
    2.87, 2.05, 2.09, 2.87, 0.52, 0.35,    # left leg
    2.87, 0.52, 2.09, 2.87, 0.52, 0.35,    # right leg
    2.09, 0.26, 0.52,                        # waist
    2.87, 2.35, 1.57, 4.71, 1.57, 0.52, 1.57,  # left arm
    2.87, 1.57, 1.57, 4.71, 1.57, 0.52, 1.57,  # right arm
], dtype=np.float64)

G1_VEL_LIMITS = np.full(29, 10.0, dtype=np.float64)  # rad/s, conservative

G1_END_EFFECTORS = {
    "left_wrist":  "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "head":        "head_link",
    "left_ankle":  "left_ankle_roll_link",
    "right_ankle": "right_ankle_roll_link",
}


def get_g1_config() -> RobotConfig:
    return RobotConfig(
        name="unitree_g1",
        urdf_path="",   # fill in Phase 2 when Pinocchio is added
        mjcf_path=str(_GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"),
        joint_names=G1_JOINT_NAMES,
        joint_lower=G1_JOINT_LOWER,
        joint_upper=G1_JOINT_UPPER,
        vel_limits=G1_VEL_LIMITS,
        end_effectors=G1_END_EFFECTORS,
        default_pose=np.zeros(29, dtype=np.float64),
        metadata={"gmr_robot_name": "unitree_g1"},
    )
```

**Action required**: Read the actual joint limits from the GMR MuJoCo XML:
```bash
grep -A2 "actuatorfrcrange\|ctrlrange" /mnt/code/GMR/assets/unitree_g1/g1_mocap_29dof.xml | head -60
```
Update `G1_JOINT_LOWER` / `G1_JOINT_UPPER` with the real values before Phase 2.

---

## Wire It

### `sensei/pipeline.py`

```python
"""
Pipeline: source → solver → metrics → output.
"""
from __future__ import annotations
from sensei.base.source import MotionSource
from sensei.base.solver import RetargetingSolver
from sensei.base.metric import Metric
from sensei.types import MotionSequence, RobotConfig, RobotMotion, MetricResult


def run(
    source: MotionSource,
    solver: RetargetingSolver,
    robot: RobotConfig,
    input_path: str,
    metrics: list[Metric] | None = None,
) -> tuple[RobotMotion, list[MetricResult]]:
    """Run the full pipeline on a single input."""
    motion: MotionSequence = source.load(input_path)
    solver.setup(robot)
    try:
        result: RobotMotion = solver.solve(motion)
    finally:
        solver.teardown()

    metric_results = []
    if metrics:
        for m in metrics:
            metric_results.append(m.compute(motion, result))

    return result, metric_results
```

### `scripts/run_pipeline.py`

```python
#!/usr/bin/env python3
"""
Run a source + solver on a single input file.

Examples:
    python scripts/run_pipeline.py \
        --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
        --source gvhmr \
        --solver gmr \
        --robot g1 \
        --output outputs/tennis_g1_gmr.pkl
"""
import argparse
import pickle
import numpy as np
from sensei.sources.gvhmr import GVHMRSource
from sensei.solvers.gmr import GMRSolver
from sensei.metrics.timing import SolverTimingMetric
from sensei.robots.g1 import get_g1_config
from sensei.pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", default="gvhmr")
    parser.add_argument("--solver", default="gmr")
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = GVHMRSource()
    solver = GMRSolver(verbose=False)
    robot = get_g1_config()
    metrics = [SolverTimingMetric()]

    result, metric_results = run(source, solver, robot, args.input, metrics)

    # Print timing report
    for mr in metric_results:
        print(f"\n=== {mr.name} ===")
        for k, v in mr.metadata.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save robot motion
    if args.output:
        root_pos = np.zeros((len(result.states), 3))
        root_rot = np.zeros((len(result.states), 4))
        dof_pos = np.array([s.q for s in result.states])
        data = {
            "fps": result.fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "solver": result.solver_name,
        }
        with open(args.output, "wb") as f:
            pickle.dump(data, f)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## Tests

### `tests/conftest.py`

```python
import pathlib
import pytest
import numpy as np
from sensei.types import MotionSequence, RobotConfig

GVHMR_TEST_FILE = "/mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt"

@pytest.fixture
def tennis_pt_path():
    return GVHMR_TEST_FILE

@pytest.fixture
def mock_motion_sequence():
    N, J = 10, 21
    return MotionSequence(
        fps=30.0,
        num_frames=N,
        num_joints=J,
        body_pose=np.zeros((N, J, 3), dtype=np.float64),
        global_orient=np.zeros((N, 3), dtype=np.float64),
        transl=np.zeros((N, 3), dtype=np.float64),
        betas=np.zeros(10, dtype=np.float64),
        joint_names=[f"joint_{i}" for i in range(J)],
        landmarks=None,
    )
```

### `tests/test_sources/test_gvhmr_source.py`

```python
import pytest
import numpy as np
from sensei.sources.gvhmr import GVHMRSource

def test_can_load(tennis_pt_path):
    src = GVHMRSource()
    assert src.can_load(tennis_pt_path)
    assert not src.can_load("nonexistent.pkl")

def test_load_returns_motion_sequence(tennis_pt_path):
    src = GVHMRSource()
    motion = src.load(tennis_pt_path)

    assert motion.fps == pytest.approx(30.0)
    assert motion.num_frames > 0
    assert motion.body_pose.dtype == np.float64
    assert motion.body_pose.shape == (motion.num_frames, motion.num_joints, 3)
    assert motion.global_orient.shape == (motion.num_frames, 3)
    assert motion.transl.shape == (motion.num_frames, 3)
    assert motion.betas.shape == (10,)
    assert motion.landmarks is not None
    assert len(motion.landmarks) == motion.num_frames
    # Each landmark dict should have at least Hips and wrist keys
    frame0 = motion.landmarks[0]
    assert "Hips" in frame0
    pos, rot = frame0["Hips"]
    assert pos.shape == (3,)
```

### `tests/test_solvers/test_gmr_solver.py`

```python
import pytest
import numpy as np
from sensei.solvers.gmr import GMRSolver
from sensei.robots.g1 import get_g1_config
from sensei.sources.gvhmr import GVHMRSource

def test_gmr_solver_short_clip(tennis_pt_path):
    src = GVHMRSource()
    motion = src.load(tennis_pt_path)

    # Only run first 30 frames to keep test fast
    motion.num_frames = 30
    motion.body_pose = motion.body_pose[:30]
    motion.global_orient = motion.global_orient[:30]
    motion.transl = motion.transl[:30]
    motion.landmarks = motion.landmarks[:30]

    robot = get_g1_config()
    solver = GMRSolver(verbose=False)
    solver.setup(robot)
    result = solver.solve(motion)
    solver.teardown()

    assert len(result.states) == 30
    assert result.states[0].q.shape == (29,)
    assert result.states[0].q.dtype == np.float64
    assert np.mean(result.converged) > 0.9   # expect >90% convergence
```

### `tests/test_metrics/test_timing_metric.py`

```python
import numpy as np
from sensei.metrics.timing import SolverTimingMetric
from sensei.types import RobotMotion, RobotState

def test_timing_metric():
    N = 50
    frame_times = np.random.uniform(0.01, 0.02, N)  # 10–20 ms per frame
    states = [RobotState(q=np.zeros(29), dq=np.zeros(29), timestamp=i/30.0) for i in range(N)]
    result = RobotMotion(
        robot_name="unitree_g1",
        solver_name="gmr",
        fps=30.0,
        states=states,
        converged=np.ones(N, dtype=bool),
        metadata={"frame_times_s": frame_times},
    )
    metric = SolverTimingMetric()
    mr = metric.compute(None, result)
    assert mr.unit == "fps"
    assert mr.value > 0
    assert "latency_p95_ms" in mr.metadata
```

### `tests/test_base/test_abc_compliance.py`

```python
"""Verify all registered sources/solvers/metrics satisfy their ABCs."""
from sensei.base.source import MotionSource
from sensei.base.solver import RetargetingSolver
from sensei.base.metric import Metric
from sensei.sources.gvhmr import GVHMRSource
from sensei.solvers.gmr import GMRSolver
from sensei.metrics.timing import SolverTimingMetric

def test_sources_are_subclasses():
    assert issubclass(GVHMRSource, MotionSource)

def test_solvers_are_subclasses():
    assert issubclass(GMRSolver, RetargetingSolver)

def test_metrics_are_subclasses():
    assert issubclass(SolverTimingMetric, Metric)

def test_sources_have_name():
    assert GVHMRSource().name == "gvhmr"

def test_solvers_have_name():
    assert GMRSolver().name == "gmr"
```

---

## Acceptance Criteria

Phase 1 is complete when all of the following pass:

- [ ] `pytest tests/` — all tests green
- [ ] `python scripts/run_pipeline.py --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt --output /tmp/tennis_g1_gmr.pkl` — runs without error
- [ ] Timing report shows FPS ≥ 30 (GMR's published baseline: 60–70 FPS on high-end CPUs)
- [ ] Output `.pkl` has `dof_pos` shape `(N, 29)` and values within `G1_JOINT_LOWER` / `G1_JOINT_UPPER`
- [ ] Convergence rate ≥ 90% on all four test clips

Run all four test clips:
```bash
for clip in tennis basketball_clip dance_clip 0_input_video; do
    python scripts/run_pipeline.py \
        --input /mnt/code/GVHMR/outputs/demo/${clip}/hmr4d_results.pt \
        --output outputs/${clip}_g1_gmr.pkl
done
```

---

## Common Pitfalls

### SMPL-X body model not found
```
FileNotFoundError: .../assets/body_models/smplx/SMPLX_NEUTRAL.npz
```
Fix: Confirm `/mnt/code/GMR/assets/body_models/smplx/` exists and contains the model files. These are not included in the GMR repo — you may need to download them from the SMPL-X project website.

### GMR not installed
```
ModuleNotFoundError: No module named 'general_motion_retargeting'
```
Fix: `pip install -e /mnt/code/GMR`

### DAQP solver not available
```
ValueError: Solver 'daqp' is not available
```
Fix: `pip install qpsolvers[daqp]`

### Wrong landmark dict format
`GMR.retarget()` expects the per-frame dict returned by `get_gvhmr_data_offline_fast`, not a raw array. Do not unpack or re-format the landmark dicts — pass them directly.

### Human height not updating
If you see unexpected scale issues, check whether `GeneralMotionRetargeting` supports updating `actual_human_height` after construction. If not, re-create the solver with the correct height per sequence.

### `body_pose` shape mismatch after FPS alignment
`smplx_data["pose_body"]` has shape `(N_orig, 63)`. After `get_gvhmr_data_offline_fast` with FPS alignment, the landmark list may have `N_aligned ≠ N_orig`. Always use `len(frames_list)` as `N`, not `smplx_data["pose_body"].shape[0]`.
