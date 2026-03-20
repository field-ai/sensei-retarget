# sensei-retarget

Modular platform for developing and benchmarking humanoid motion retargeting pipelines.

**Target robot**: Unitree G1 (29-DoF)
**Upstream sources**: GVHMR video inference, SMPL-X mocap, BVH (planned)
**Solvers**: GMR/mink (Phase 1), Pinocchio+CasADi+IPOPT (Phase 2), alpaqa PANOC (Phase 3)

---

## Quickstart

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate sensei

# 2. Install third-party libraries we import from
pip install -e third_party/GMR      # GMR + mink + mujoco + smplx

# 3. Install sensei
pip install -e .

# 4. Run the Phase 1 baseline on a pre-processed clip
python scripts/run_pipeline.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --source gvhmr --solver gmr --robot g1

# 5. Run tests (no solver deps required)
pytest tests/test_base/
```

---

## Architecture

```
upstream source  →  MotionSequence  →  solver  →  RobotMotion  →  metrics
 (GVHMRSource)                       (GMRSolver)              (SolverTimingMetric)
```

Three abstract base classes define the contracts:

| ABC | File | What it produces |
|-----|------|-----------------|
| `MotionSource` | `sensei/base/source.py` | `MotionSequence` from a file |
| `RetargetingSolver` | `sensei/base/solver.py` | `RobotMotion` from a `MotionSequence` |
| `Metric` | `sensei/base/metric.py` | `MetricResult` from source + result |

All arrays are `float64` numpy. Solver dependencies are late-imported so missing a solver's deps doesn't break other solvers.

---

## Repository layout

```
sensei/             Main package
  base/             ABCs (the stable contract layer)
  types.py          Shared data types (MotionSequence, RobotMotion, …)
  sources/          Upstream data adapters
  solvers/          Retargeting/IK solvers (one file each)
  metrics/          Evaluation metrics
  robots/           Robot configurations (G1, …)
  registry.py       Auto-discovery of sources/solvers/metrics

scripts/            CLI entry points
tests/              Unit + integration tests

third_party/        Repos we import from — gitignored, not pushed
  GMR/              → /mnt/code/GMR (symlink)
  GVHMR/            → /mnt/code/GVHMR (symlink)
  alpaqa/           Phase 3 optimizer

reference_repos/    Read-only inspiration — gitignored, not pushed
  mink/ G1Pilot/ xr_teleoperate/ pinocchio-casadi-examples/ …

docs/
  plan.md           Master plan and phase roadmap
  phase1.md         Detailed Phase 1 build guide
```

---

## Phase roadmap

| Phase | Solver | Status |
|-------|--------|--------|
| 0 | Package scaffold + ABCs | ✅ done |
| 1 | GVHMR + GMR (mink QP IK) | 🔧 in progress |
| 2 | Pinocchio + CasADi + IPOPT | planned |
| 3 | Pinocchio + alpaqa PANOC | planned |
| 4 | Full metrics suite | planned |

See [docs/plan.md](docs/plan.md) for full detail.

---

## Adding a new solver

1. Create `sensei/solvers/<name>.py`, subclass `RetargetingSolver`
2. Import solver deps inside `setup()` (not at module top-level)
3. Register: `registry.register_solver(MySolver)` at module level
4. Add `configs/solvers/<name>.yaml`
5. Add test in `tests/test_solvers/`
6. Run `pytest tests/test_base/test_abc_compliance.py`

See [AGENTS.md](AGENTS.md) for the step-by-step agent workflow.
