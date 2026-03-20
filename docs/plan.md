# Sensei Retarget — Master Plan

> **Purpose**: A modular, benchmarkable platform for developing and testing humanoid motion retargeting pipelines. Target robot: Unitree G1. Designed for agentic (vibe) coding.

---

## Table of Contents

1. [Vision & Goals](#vision--goals)
2. [Environment Setup](#environment-setup)
3. [Third-Party Library Strategy](#third-party-library-strategy)
4. [Repository Architecture](#repository-architecture)
5. [Standard Data Types](#standard-data-types)
6. [Abstract Base Classes](#abstract-base-classes)
7. [Phase Plan](#phase-plan)
8. [Reference Ecosystem](#reference-ecosystem)
9. [Plan: AGENTS.md](#plan-agentsmd)
10. [Plan: CLAUDE.md](#plan-claudemd)

---

## Vision & Goals

Build a research platform where:
- Any **upstream motion source** (video, mocap, bespoke model) plugs in via a common interface
- Any **IK/retargeting solver** is swappable and benchmarkable via a common interface
- Any **metric** (timing, accuracy, smoothness) registers and runs across solver × source combinations
- The G1 humanoid is the primary target, architecture open to other robots
- Solver dependencies are **fully isolated** — installing one solver does not affect another
- The codebase is clean enough for an AI agent to navigate, extend, and test autonomously

**Non-goals (for now)**: RL fine-tuning, sim-to-real transfer, hardware deployment.

---

## Environment Setup

All development uses a single conda environment (`sensei`). The env grows phase by phase — you activate it once and add deps as you reach each phase.

### Files

| File | Purpose |
|------|---------|
| `environment.yml` | Conda env definition (Phase 1 active, Phase 2/3 commented) |
| `setup.py` | Python package definition; `pip install -e .` installs `sensei` into the env |

### Phase 1 — create and activate

```bash
# 1. Create the conda env (Python 3.10, numpy, scipy, torch, mink, mujoco, smplx)
conda env create -f environment.yml

# 2. Activate
conda activate sensei

# 3. Install third-party repos (in order — GMR depends on mink already installed above)
pip install -e third_party/GMR       # installs general_motion_retargeting
pip install -e third_party/GVHMR    # optional: only needed to run GVHMR inference
                                     # loading .pt files works without this

# 4. Install sensei itself (editable)
pip install -e .

# 5. Verify
python -c "from sensei.sources.gvhmr import GVHMRSource; print('ok')"
```

### Phase 2 — add Pinocchio + CasADi + IPOPT

```bash
# Uncomment the pinocchio/casadi/ipopt lines in environment.yml, then:
conda env update -f environment.yml --prune
```

Why conda for these: `pinocchio >= 3.0` requires C++ build tools and the `pinocchio.casadi` symbolic sub-module, which the conda-forge binary bundles correctly. `pip install pin` works but the casadi bridge is unreliable.

### Phase 3 — add OpEn

```bash
# Requires Rust toolchain (https://rustup.rs)
pip install opengen
```

### GPU vs CPU note

`environment.yml` defaults to `pytorch::cpuonly`. This is sufficient for Phase 1 (loading GVHMR `.pt` files doesn't need CUDA). To run full GVHMR inference (Phase 5), swap for the CUDA build:

```bash
# In environment.yml: remove cpuonly line, add:
#   - pytorch::pytorch=2.3.0=*cuda121*
#   - pytorch::torchvision=0.18.0=*cuda121*
# Then: conda env update -f environment.yml
```

---

## Third-Party Library Strategy

### Two tiers of external repos

**`third_party/`** — repos we **import from**. These are pip-installed into the environment and referenced by code. They are **gitignored** (not committed, not pushed). Populate by following `CLAUDE.md` setup instructions.

| Repo | Local path | How populated | Used by |
|------|-----------|--------------|---------|
| GMR | `third_party/GMR` → `/mnt/code/GMR` | symlink (exists at `/mnt/code/GMR`) | `GVHMRSource`, `GMRSolver` |
| GVHMR | `third_party/GVHMR` → `/mnt/code/GVHMR` | symlink (exists at `/mnt/code/GVHMR`) | `GVHMRSource` |
| opengen | pip package | `pip install opengen` (+ Rust toolchain) | `PinocchioOpEnSolver` (Phase 3) |

New third-party deps are added here as needed. Pinocchio and CasADi are system/conda packages, not cloned.

**`reference_repos/`** — repos we **read for inspiration**. Never imported. Also gitignored.

```
reference_repos/
├── G1Pilot/            # G1 Pinocchio+CasADi IK reference
├── mink/               # Differential IK library (used by GMR)
├── xr_teleoperate/     # G1 reduced-robot + teleoperation
├── pinocchio-casadi-examples/
├── OmniRetarget/       # Whole-body retargeting + RL reference
├── relaxed_ik_core/    # Collision-aware IK reference
└── sam-3d-body/        # Single-image 3D body reference
```

### Python dependency extras (pyproject.toml)

**Tier 1 — Core** (always installed, no solver deps):
```
numpy >= 1.24        # float64 arrays throughout
scipy >= 1.10        # rotations, spatial transforms
```
`sensei/base/` and `sensei/types.py` import only these. Never solver-specific packages.

**Tier 2 — Solver optional extras**:

| Extra | Libraries | Install method | Notes |
|-------|-----------|---------------|-------|
| `[gmr]` | mink, mujoco, qpsolvers[daqp,proxqp], smplx, torch | pip | Phase 1 |
| `[pinocchio]` | pinocchio >= 3.0, casadi >= 3.6 | **conda-forge** preferred | Phase 2/3 |
| `[open]` | opengen | `pip install opengen` | Phase 3 |
| `[ipopt]` | cyipopt | conda-forge | Phase 2 |

### Install notes
- **Pinocchio + CasADi**: `conda install -c conda-forge pinocchio casadi`. `pip install pin` is possible but lacks the `pinocchio.casadi` symbolic module.
- **IPOPT**: `conda install -c conda-forge ipopt cyipopt`
- **GMR**: `pip install -e third_party/GMR` (installs mink, mujoco, smplx as transitive deps)
- **GVHMR**: `pip install -e third_party/GVHMR` (requires CUDA only for running inference; loading `.pt` files is CPU-only)
- **OpEn**: `pip install opengen` (Rust toolchain required — `curl https://sh.rustup.rs -sSf | sh`)

### Solver isolation rule
Each solver file imports its dependencies **inside the class**, not at module top-level:
```python
# sensei/solvers/pinocchio_ipopt.py
class PinocchioIPOPTSolver(RetargetingSolver):
    def setup(self, robot: RobotConfig) -> None:
        import pinocchio as pin          # late import — only fails if not installed
        from pinocchio import casadi as cpin
        import casadi
        ...
```
`sensei/registry.py` attempts a test import for each solver at startup and skips unavailable ones with a clear warning.

---

## Repository Architecture

```
sensei-humanoid-retarget/
├── CLAUDE.md                        # Human developer guide
├── AGENTS.md                        # Agent navigation guide
├── README.md                        # Project overview + quickstart
├── pyproject.toml                   # Modern packaging (PEP 517/518), optional extras
│
├── docs/
│   ├── plan.md                      # This file
│   ├── phase1.md                    # Detailed Phase 1 build guide  ← see this
│   ├── architecture.md              # Data flow, class diagrams
│   └── references.md                # Annotated bibliography
│
├── sensei/                          # Main installable package
│   ├── __init__.py
│   │
│   ├── base/                        # Stable contract layer — change with care
│   │   ├── __init__.py
│   │   ├── source.py                # MotionSource ABC
│   │   ├── solver.py                # RetargetingSolver ABC
│   │   └── metric.py                # Metric ABC
│   │
│   ├── types.py                     # MotionSequence, RobotMotion, MetricResult
│   │                                # Uses only numpy + stdlib — no solver deps
│   │
│   ├── sources/                     # Upstream data adapters
│   │   ├── __init__.py
│   │   ├── gvhmr.py                 # GVHMR .pkl → MotionSequence
│   │   ├── smplx.py                 # Raw SMPL-X file → MotionSequence
│   │   └── bvh.py                   # BVH mocap → MotionSequence
│   │
│   ├── solvers/                     # One file per solver — each self-contained
│   │   ├── __init__.py
│   │   ├── gmr.py                   # GMR + mink (Phase 1)
│   │   ├── pinocchio_ipopt.py       # Pinocchio + CasADi + IPOPT (Phase 2)
│   │   └── pinocchio_open.py        # Pinocchio + OpEn PANOC (Phase 3)
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── timing.py                # FPS, per-frame latency (p50/p95/p99)
│   │   ├── accuracy.py              # End-effector position + orientation error
│   │   ├── smoothness.py            # Joint velocity / acceleration norms
│   │   ├── joint_limits.py          # % frames violating joint limits
│   │   └── convergence.py           # % frames where solver converged
│   │
│   ├── robots/
│   │   ├── __init__.py
│   │   └── g1.py                    # G1 config: URDF path, joint names/limits
│   │
│   ├── pipeline.py                  # Orchestrator: source → solver → metrics
│   └── registry.py                  # Auto-discover + register sources/solvers/metrics
│
├── scripts/
│   ├── run_pipeline.py              # CLI: run a source+solver combo
│   ├── benchmark.py                 # CLI: solver × source comparison table
│   └── visualize.py                 # CLI: replay RobotMotion in MuJoCo viewer
│
├── configs/                         # YAML configs (Hydra-compatible)
│   ├── default.yaml
│   ├── sources/
│   │   ├── gvhmr.yaml
│   │   └── smplx.yaml
│   ├── solvers/
│   │   ├── gmr.yaml
│   │   ├── pinocchio_ipopt.yaml
│   │   └── pinocchio_open.yaml
│   └── robots/
│       └── g1.yaml
│
├── tests/
│   ├── conftest.py                  # Fixtures: sample GVHMR .pkl, mock RobotConfig
│   ├── test_base/
│   │   └── test_abc_compliance.py   # Every registered impl must satisfy its ABC
│   ├── test_sources/
│   │   └── test_gvhmr_source.py
│   ├── test_solvers/
│   │   └── test_gmr_solver.py
│   └── test_metrics/
│       └── test_timing_metric.py
│
├── third_party/                     # Repos we import from — gitignored, not pushed
│   ├── GMR -> /mnt/code/GMR         # symlink — pip install -e third_party/GMR
│   ├── GVHMR -> /mnt/code/GVHMR    # symlink — pip install -e third_party/GVHMR
│   └── (opengen is a pip package, not cloned)
│
├── reference_repos/                 # Read-only inspiration — gitignored, not pushed
│   ├── mink/
│   ├── G1Pilot/
│   ├── xr_teleoperate/
│   ├── pinocchio-casadi-examples/
│   ├── OmniRetarget/
│   ├── relaxed_ik_core/
│   └── sam-3d-body/
│
├── .gitignore                       # excludes third_party/, reference_repos/, data/, outputs/
│
└── data/                            # NOT git-tracked
    └── README.md                    # Documents expected data layout
```

---

## Standard Data Types

Based on the reference ecosystem survey:

### dtype: `float64` throughout
- mink uses `np.float64` exclusively in all lie algebra and QP operations
- Pinocchio defaults to `float64` (Eigen::VectorXd)
- GMR uses default numpy (float64)
- xr_teleoperate `G1_29_ArmIK` uses `np.float64` for all joint arrays

Only exception: GVHMR outputs `torch.float32` tensors — convert to `float64 numpy` in the source adapter.

### Transform convention: rotation matrices or axis-angle, not quaternions
- Pinocchio uses `pin.SE3` (4×4 or SO3 rotation matrix + translation)
- mink uses `mink.SE3` (same convention)
- GVHMR outputs axis-angle (`(N, 3)` per joint)
- `scipy.spatial.transform.Rotation` is the bridge (handles all conversions)

### `MotionSequence` — canonical intermediate type

```python
# sensei/types.py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class MotionSequence:
    """
    Canonical representation between any source and any solver.
    All arrays are float64 numpy. Rotation is axis-angle (N, J, 3).
    """
    fps: float
    num_frames: int                  # N
    num_joints: int                  # J (22 for SMPL body)
    body_pose: np.ndarray            # (N, J, 3) axis-angle per joint, float64
    global_orient: np.ndarray        # (N, 3) root orientation axis-angle, float64
    transl: np.ndarray               # (N, 3) root translation in metres, float64
    betas: np.ndarray                # (10,) body shape params, float64
    joint_names: list[str]           # length J, canonical SMPL names
    metadata: dict = field(default_factory=dict)

@dataclass
class RobotConfig:
    name: str                        # e.g. "unitree_g1"
    urdf_path: str
    joint_names: list[str]           # ordered DoF names
    joint_lower: np.ndarray          # (DoF,) float64
    joint_upper: np.ndarray          # (DoF,) float64
    vel_limits: np.ndarray           # (DoF,) float64, rad/s
    end_effectors: dict[str, str]    # label -> link name, e.g. {"left_wrist": "left_wrist_roll_link"}

@dataclass
class RobotState:
    q: np.ndarray                    # (DoF,) joint positions, float64
    dq: np.ndarray                   # (DoF,) joint velocities, float64
    timestamp: float                 # seconds

@dataclass
class RobotMotion:
    robot_name: str
    solver_name: str
    fps: float
    states: list[RobotState]
    converged: np.ndarray            # (N,) bool, did solver converge this frame?
    metadata: dict = field(default_factory=dict)

@dataclass
class MetricResult:
    name: str
    value: float
    unit: str
    per_frame: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
```

No torch tensors in these types. Sources are responsible for converting from their native format.

---

## Abstract Base Classes

### `MotionSource` (`sensei/base/source.py`)

```python
from abc import ABC, abstractmethod
from sensei.types import MotionSequence

class MotionSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, path: str) -> MotionSequence:
        """Load file at `path`, return canonical MotionSequence."""
        ...

    @abstractmethod
    def can_load(self, path: str) -> bool:
        """Return True if this source can handle the given path."""
        ...
```

### `RetargetingSolver` (`sensei/base/solver.py`)

```python
from abc import ABC, abstractmethod
from sensei.types import MotionSequence, RobotConfig, RobotMotion
import numpy as np

class RetargetingSolver(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def setup(self, robot: RobotConfig) -> None:
        """Initialize solver. Called once before any solve calls."""
        ...

    @abstractmethod
    def solve(self, motion: MotionSequence) -> RobotMotion:
        """Batch solve all frames. Default impl calls solve_frame() in a loop."""
        ...

    @abstractmethod
    def solve_frame(self, targets: dict[str, np.ndarray], q_prev: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Solve a single frame.
        Args:
            targets: {ee_label: SE3 (4x4 float64)} end-effector targets
            q_prev:  (DoF,) previous joint configuration for warm-start
        Returns:
            q: (DoF,) solution
            converged: bool
        """
        ...

    def teardown(self) -> None:
        """Optional cleanup. Called when pipeline is done."""
        pass
```

### `Metric` (`sensei/base/metric.py`)

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
    def compute(self, source: MotionSequence, result: RobotMotion) -> MetricResult:
        ...
```

---

## Phase Plan

### Phase 0 — Scaffold

- [x] `environment.yml` — conda env (Phase 1 active; Phase 2/3 commented)
- [x] `setup.py` — Python package; `pip install -e .`
- [x] `.gitignore` — excludes `third_party/`, `reference_repos/`, `data/`, `outputs/`
- [ ] `sensei/` package skeleton with all `__init__.py` files
- [ ] `sensei/base/` — `MotionSource`, `RetargetingSolver`, `Metric` ABCs
- [ ] `sensei/types.py` — `MotionSequence`, `RobotConfig`, `RobotMotion`, `MetricResult`
- [ ] `sensei/registry.py` — source/solver/metric registration
- [ ] `sensei/robots/g1.py` — G1 `RobotConfig` with joint names and limits
- [ ] `tests/test_base/test_abc_compliance.py`
- [ ] `CLAUDE.md`, `AGENTS.md`, `README.md`

---

### Phase 1 — GVHMR + GMR Baseline

**See [docs/phase1.md](phase1.md) for full detail.**

Summary:
- Source: `GVHMRSource` reads `.pkl` files output by GVHMR
- Solver: `GMRSolver` wraps `GeneralMotionRetargeting` from `/mnt/code/GMR`
- Robot: G1 (`unitree_g1` in GMR assets)
- Metric: `SolverTimingMetric` (FPS, p50/p95/p99 latency)
- End goal: reproduce GMR's ~60 FPS on the already-processed test clips

---

### Phase 2 — Pinocchio + CasADi + IPOPT

**See [docs/phase2.md](phase2.md) for full detail.**

Two sub-phases sharing one solver class (`PinocchioIPOPTSolver`, toggled by `collision=True/False`):

**Phase 2a — Kinematic NLP**
- Solver: `PinocchioIPOPTSolver(collision=False)`
- Load G1 URDF, `buildReducedRobot()` — lock legs + waist (14 active arm DoF)
- `cpin.Model` symbolic copy → CasADi `Opti` NLP
- Cost: EE trans error (×50) + rotation log3 error (×3) + regularisation + smoothing
- Constraints: joint position limits only
- IPOPT with warm start (`acceptable_iter=5` for early exit)
- Add `AccuracyMetric` (EE position error mm, orientation error deg)

**Phase 2b — Collision-Aware NLP**
- Solver: `PinocchioIPOPTSolver(collision=True)`
- Ground constraint: symbolic FK foot height ≥ 0 (exact, CasADi autodiff)
- Self-collision: sphere approximations for key body pairs (wrists, elbows, torso, pelvis) — squared distance constraints, fully symbolic
- Future: linearised FCL mesh-accurate constraints (Phase 3+)

---

### Phase 3 — OpEn (Optimization Engine) PANOC Solver

**See [docs/phase3.md](phase3.md) for full detail.**

- Solver: `PinocchioOpEnSolver`
- Reuse Phase 2 Pinocchio model + CasADi symbolic expressions for cost/constraints
- `opengen` takes the CasADi NLP and **generates a compiled Rust solver** (offline, at `setup()`)
- At runtime, call the pre-compiled solver via local TCP socket → expected sub-ms solve times
- PANOC (proximal averaged Newton-type) + Augmented Lagrangian for constraints — no matrix factorizations
- Benchmark: FPS, accuracy, smoothness vs. Phase 1 (GMR) and Phase 2 (IPOPT)

---

### Phase 4 — Metrics Expansion

Full suite:

| Metric | Unit | Implementation |
|--------|------|---------------|
| Solver FPS | FPS | wall-clock / num_frames |
| Frame latency p50/p95/p99 | ms | per-frame timing array |
| EE position error | mm | FK(q) vs. target, Euclidean |
| EE orientation error | deg | geodesic distance on SO(3) |
| Joint velocity norm | rad/s | finite diff on q sequence |
| Joint acceleration norm | rad/s² | second-order FD |
| Joint limit violations | % frames | count(q outside limits) / N |
| Convergence rate | % frames | sum(converged) / N |

`scripts/benchmark.py` produces a Markdown table: solver × metric.

---

### Phase 5 — Additional Sources (future)

- `VideoSource`: wraps GVHMR inference on-the-fly from raw video
- `SMPLXSource`: load AMASS/OMOMO `.pkl` directly
- `BVHSource`: reuse GMR's BVH parser
- `SAM3DBodySource`: single-image upstream (per-frame, no temporal)

---

## Reference Ecosystem

### Third-party (imported, `third_party/`)

| Repo | Local path | Role |
|------|-----------|------|
| GMR | `third_party/GMR` | Phase 1 solver (mink IK); also SMPL-X utilities |
| GVHMR | `third_party/GVHMR` | Upstream: video → SMPL-X `.pt` files |
| opengen | pip package | Phase 3 OpEn PANOC code generator |

### Reference repos (read-only, `reference_repos/`)

| Repo | Local path | Key insights |
|------|-----------|------|
| mink | `reference_repos/mink` | Differential IK API: `solve_ik()`, `FrameTask`, float64 |
| G1Pilot | `reference_repos/G1Pilot` | G1 Pinocchio+CasADi IK: locked-joint reduction, cost formulation |
| xr_teleoperate | `reference_repos/xr_teleoperate` | G1 `G1_29_ArmIK` class: `pin.RobotWrapper`, cache pattern |
| pinocchio-casadi-examples | `reference_repos/pinocchio-casadi-examples` | OCP template with `pinocchio.casadi` + CasADi `Opti` |
| OmniRetarget | `reference_repos/OmniRetarget` | Whole-body retargeting + RL pipeline reference |
| relaxed_ik_core | `reference_repos/relaxed_ik_core` | Collision-aware IK reference (Rust) |
| sam-3d-body | `reference_repos/sam-3d-body` | Single-image upstream source (future) |

---

## Plan: AGENTS.md

Write at repo root after Phase 0. Sections:

1. **Repo map** — annotated tree, what to read first
2. **How to add a Source** — subclass `MotionSource`, register, add YAML, add test
3. **How to add a Solver** — subclass `RetargetingSolver`, late-import deps, register, add YAML, add test
4. **How to add a Metric** — subclass `Metric`, register, add test
5. **How to run a benchmark** — exact CLI, expected output
6. **Key invariants** — ABCs are stable contracts; `types.py` is backward-compatible; all impls pass `test_abc_compliance.py`
7. **Data layout** — where to find/put test data, what's gitignored
8. **Testing** — how to run, what CI checks exist

---

## Plan: CLAUDE.md

Write at repo root after Phase 0. Sections:

1. **What this repo does** (3 sentences)
2. **Dev environment setup** — exact commands per optional extra
3. **Run commands** — copy-pasteable one-liners for every operation
4. **Architecture decisions** — why ABCs, why `MotionSequence`, why Hydra, why `solve()` + `solve_frame()`
5. **Phase-by-phase build guide** — pointer to current phase doc
6. **Dependency install quirks** — Pinocchio conda, IPOPT conda-forge, GVHMR path-install
7. **Data layout** — symlinks, gitignore
8. **Common failure modes** — URDF paths, SMPL-X model paths, CUDA OOM in GVHMR

---

## Immediate Next Step

Phase 0 scaffold. Start by reading `docs/phase1.md`.
