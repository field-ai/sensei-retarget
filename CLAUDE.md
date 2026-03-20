# CLAUDE.md — Developer Guide

This file tells Claude Code (and human developers) how to build and work in this repo.

---

## What this repo does

`sensei-retarget` is a modular benchmark platform for humanoid motion retargeting pipelines. It takes upstream human motion data (video, mocap) and retargets it to the Unitree G1 robot using swappable IK solvers. Each solver can be benchmarked against the same metrics.

---

## Environment setup

```bash
# Create the conda env (Python 3.10, numpy, scipy, torch, mink, mujoco, smplx)
conda env create -f environment.yml
conda activate sensei

# Install third-party repos we import from (order matters)
pip install -e third_party/GMR      # installs GMR + mink + mujoco + smplx
pip install -e third_party/GVHMR   # optional — only needed to run GVHMR inference

# Install sensei itself
pip install -e .
```

### Phase 2 additions (Pinocchio + CasADi + IPOPT)

```bash
conda install -c conda-forge pinocchio casadi ipopt
```

Why conda for pinocchio: the `pinocchio.casadi` symbolic sub-module requires the C++ build from conda-forge. `pip install pin` works but the CasADi bridge is unreliable.

### Phase 3 additions (alpaqa)

```bash
pip install -e third_party/alpaqa
```

### GPU note

`environment.yml` defaults to CPU-only torch. This is sufficient for all of Phase 1 (loading GVHMR `.pt` files doesn't need CUDA). To run GVHMR inference, swap the pytorch line for the CUDA build.

---

## Run commands

```bash
# Run Phase 1 pipeline — GMR solver
python scripts/run_pipeline.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --source gvhmr --solver gmr --robot g1

# Run Phase 2a — Pinocchio + IPOPT (full 29-DoF)
python scripts/run_pipeline.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --source gvhmr --solver pinocchio_ipopt --robot g1

# Timing + accuracy metrics (GMR vs Phase 2a side-by-side)
python scripts/plot_metrics.py

# Run tests (ABC compliance — no solver deps needed)
pytest tests/test_base/

# Run all tests (requires sensei conda env + GMR + pinocchio installed)
pytest tests/

# Lint
ruff check sensei/ tests/
```

---

## Architecture decisions

**Why ABCs instead of protocols?**
ABCs enforce the `setup()` → `solve()` call ordering via runtime checks. Protocols would require manual documentation of this lifecycle.

**Why `MotionSequence.landmarks`?**
GMR's `retarget()` expects pre-computed SMPL-X FK output (body landmark positions + orientations). Running FK in the source keeps solvers clean — they receive Cartesian targets regardless of whether the source was GVHMR, BVH, or something else.

**Why late imports in solvers?**
`import pinocchio` at module top-level would crash on machines without it installed. Late imports (inside `setup()`) mean `from sensei.solvers.gmr import GMRSolver` always works; the error is deferred to when you actually try to use the solver.

**Why `solve()` + `solve_frame()`?**
`solve()` handles batch processing and timing. `solve_frame()` is the minimal unit for streaming/real-time use. The base class `solve()` calls `solve_frame()` in a loop and records per-frame timing automatically.

**Why `float64` everywhere?**
mink, Pinocchio, and xr_teleoperate all use float64. GVHMR outputs float32 tensors — the source converts at load time. Mixing dtypes inside solvers causes subtle QP conditioning bugs.

---

## Phase-by-phase build guide

| Phase | Doc | Key files |
|-------|-----|-----------|
| 0 (done) | — | `sensei/base/`, `sensei/types.py`, `sensei/robots/g1.py` |
| 1 (done) | [docs/phase1.md](docs/phase1.md) | `sensei/sources/gvhmr.py`, `sensei/solvers/gmr.py`, `scripts/run_pipeline.py` |
| 2a | [docs/phase2.md](docs/phase2.md) | `sensei/solvers/pinocchio_ipopt.py`, `sensei/metrics/accuracy.py` |
| 2b | [docs/phase2.md](docs/phase2.md) | same solver, `collision=True` (ground + self-collision) |
| 3 | — | `sensei/solvers/pinocchio_alpaqa.py` |

Full roadmap: [docs/plan.md](docs/plan.md).

---

## Data paths

```
GVHMR test clips:  /mnt/code/GVHMR/outputs/demo/{tennis,basketball_clip,dance_clip,0_input_video}/hmr4d_results.pt
SMPL-X body model: /mnt/code/GMR/assets/body_models/smplx/
G1 MuJoCo XML:     /mnt/code/GMR/assets/unitree_g1/g1_mocap_29dof.xml
G1 URDF:           reference_repos/xr_teleoperate/assets/g1/g1_body29_hand14.urdf
```

`data/` is gitignored. See `data/README.md`.

---

## Common failure modes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: general_motion_retargeting` | GMR not installed | `pip install -e third_party/GMR` |
| `FileNotFoundError: SMPLX_NEUTRAL.npz` | SMPL-X body model missing | Download from smpl-x.is.tue.mpg.de, place in `/mnt/code/GMR/assets/body_models/smplx/` |
| `ValueError: Solver 'daqp' is not available` | DAQP QP backend missing | `pip install qpsolvers[daqp]` |
| `AssertionError: Call setup() before solve()` | `solver.setup(robot)` not called | Always call `setup()` before `solve()` |
| `AssertionError: GMRSolver requires .landmarks` | Source didn't run SMPL-X FK | Use `GVHMRSource`, not a bare `MotionSequence` |
| Body pose shape mismatch after FPS alignment | N from FPS alignment ≠ N_orig | Always use `len(frames_list)` as N, not `smplx_data['pose_body'].shape[0]` |
| `ModuleNotFoundError: pinocchio` | Pinocchio not installed | `conda install -c conda-forge pinocchio casadi ipopt` |
| `import pinocchio.casadi` fails | pip-installed pin, not conda | Reinstall via conda-forge — pip build lacks the CasADi bridge |
| IPOPT `Infeasible_Problem_Detected` | NLP is infeasible (joint limits too tight, or bad warm-start) | Check q_prev is within limits; increase `max_iter` |

---

## Key invariants — do not break

1. `sensei/base/` and `sensei/types.py` have **no solver-specific imports**
2. All `MotionSequence` / `RobotMotion` arrays are `float64` numpy
3. Every solver imports its heavy deps inside `setup()`, not at module top-level
4. `test_abc_compliance.py` must pass without any solver deps installed
5. `third_party/` and `reference_repos/` are never committed
