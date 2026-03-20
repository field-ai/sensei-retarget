# sensei-humanoid-retarget

**Modular benchmark platform for humanoid motion retargeting.**
Takes video-inferred human motion and retargets it to the Unitree G1 (29-DoF) using swappable IK solvers, all measured against the same metrics.

```
video  →  GVHMR  →  SMPL-X landmarks  →  [solver]  →  G1 joint angles  →  metrics
```

---

## Demo

> **GMR solver** · Unitree G1 29-DoF · clip: `0_input_video` · 30 fps
> Left: SMPL-X body model (source)   Right: Unitree G1 retargeted output

<video src="docs/assets/demo_gmr_unitree_g1.mp4" width="964" controls loop muted>
  <a href="docs/assets/demo_gmr_unitree_g1.mp4">docs/assets/demo_gmr_unitree_g1.mp4</a>
</video>

---

## Solvers

| Solver | Algorithm | Mean latency | FPS | Status |
|--------|-----------|-------------|-----|--------|
| `gmr` | GMR / mink differential QP | ~4 ms | ~250 | ✅ Phase 1 |
| `pinocchio_ipopt` | Pinocchio + CasADi + IPOPT NLP | ~18 ms | ~50 | ✅ Phase 2a |
| `pinocchio_alpaqa` | Pinocchio + alpaqa PANOC | — | — | planned Phase 3 |

> **Phase 2a note — local minima.**
> `pinocchio_ipopt` solves the full NLP each frame independently (warm-started from the previous frame). Unlike `gmr`'s differential QP, which penalises large joint-angle changes by construction and naturally path-follows, IPOPT can converge to a different local minimum — especially in unconstrained yaw joints (`waist_yaw`, `left_shoulder_yaw`). The divergence is ~1.5–1.9 rad on those joints. The robot tracks the correct end-effector positions but the arm/torso posture differs from GMR.
> Planned fix in Phase 3: switch to PANOC (alpaqa), which supports warm-starting from a local tangent and has better basin-of-attraction properties.

---

## Quickstart

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate sensei

# 2. Install third-party libraries
pip install -e third_party/GMR           # GMR + mink + mujoco + smplx
pip install -e third_party/GVHMR         # optional — GVHMR inference only

# 3. Phase 2 additions (Pinocchio + CasADi + IPOPT)
conda install -c conda-forge pinocchio casadi ipopt

# 4. Install sensei itself
pip install -e .

# 5. Run a retargeting pipeline
python scripts/run_pipeline.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --source gvhmr --solver gmr --robot g1

python scripts/run_pipeline.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --source gvhmr --solver pinocchio_ipopt --robot g1

# 6. Side-by-side metrics comparison (all clips, all solvers)
python scripts/plot_metrics.py           # → outputs/metrics_timing.png

# 7. Render a MuJoCo video
python scripts/make_video.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --solver gmr

# 8. Run tests
pytest tests/test_base/                  # no solver deps required
pytest tests/                            # full suite — needs sensei env
```

---

## Architecture

Three abstract base classes define the contracts:

| ABC | File | Contract |
|-----|------|---------|
| `MotionSource` | `sensei/base/source.py` | Load a file → `MotionSequence` |
| `RetargetingSolver` | `sensei/base/solver.py` | `MotionSequence` → `RobotMotion` |
| `Metric` | `sensei/base/metric.py` | Source + result → `MetricResult` |

Key invariants:
- **All arrays are `float64` numpy.** GVHMR emits `float32` tensors; `GVHMRSource` converts at load time.
- **Solver deps are late-imported** (inside `setup()`). Missing a solver's deps doesn't break other solvers or the test suite.
- **`test_abc_compliance.py` must always pass** without any solver deps installed.

Lifecycle every solver must follow: `setup(robot)` → `solve(motion)` → `teardown()`.

---

## Repository layout

```
sensei/
  base/           ABCs — the stable contract layer
  types.py        Shared data types (MotionSequence, RobotMotion, …)
  sources/        Upstream data adapters (GVHMRSource, …)
  solvers/        IK solvers, one file each
    gmr.py          Phase 1 — GMR / mink differential QP
    pinocchio_ipopt.py  Phase 2a — Pinocchio + CasADi + IPOPT NLP
  metrics/        Evaluation metrics
  robots/         Robot configs (G1, …)
  registry.py     Auto-discovery of sources / solvers / metrics

scripts/
  run_pipeline.py   Main CLI entry point
  plot_metrics.py   Side-by-side solver comparison chart
  make_video.py     MuJoCo render → mp4

tests/
  test_base/      ABC compliance tests (no solver deps)
  test_solvers/   Solver integration tests

docs/
  plan.md         Master plan and phase roadmap
  phase1.md       Phase 1 detailed build notes
  phase2.md       Phase 2 detailed build notes

third_party/      Repos imported at runtime — gitignored
  GMR/            → /mnt/code/GMR
  GVHMR/          → /mnt/code/GVHMR (optional)

reference_repos/  Read-only source material — gitignored
```

---

## Phase 1 results — GMR baseline

Solve time only (GVHMR pre-processing excluded). Unitree G1, 29 DoF, DAQP backend.

| Clip | Mean latency | p95 | FPS | Converged | Joint violations |
|------|-------------|-----|-----|-----------|-----------------|
| Basketball | 4.1 ms | 7.1 ms | ~248 | 100% | 4.3% |
| Dance | 5.3 ms | 6.6 ms | ~192 | 100% | 0% |
| Tennis | 3.4 ms | 4.8 ms | ~300 | 100% | 0% |

6–10× real-time headroom. The DAQP QP solve itself is ~0.06 ms/call; overhead is in mink's QP assembly and uncached MuJoCo name lookups.

---

## Phase 2a results — Pinocchio + IPOPT

Full 29-DoF NLP, warm-started per frame. Unitree G1, 29 DoF, IPOPT backend.

| Clip | Mean latency | p95 | FPS | Converged | Joint violations |
|------|-------------|-----|-----|-----------|-----------------|
| Basketball | ~18 ms | — | ~50 | 100% | 0% |
| Dance | ~20 ms | — | ~50 | 100% | 0% |
| Tennis | ~17 ms | — | ~52 | 100% | 0% |

IPOPT respects URDF joint limits exactly (box constraints in the NLP). The 0% violation rate vs GMR's occasional violations is because GMR uses softer limit handling in the QP.

See the [local minima note](#solvers) above for posture divergence.

---

## Phase roadmap

| Phase | Solver | Status |
|-------|--------|--------|
| 0 | Package scaffold + ABCs | ✅ done |
| 1 | GVHMR source + GMR solver | ✅ done |
| 2a | Pinocchio + CasADi + IPOPT | ✅ done |
| 2b | Phase 2a + collision avoidance | in progress |
| 3 | Pinocchio + alpaqa PANOC | planned |
| 4 | Full metrics suite + leaderboard | planned |

Full detail: [docs/plan.md](docs/plan.md).

---

## Adding a new solver

1. Create `sensei/solvers/<name>.py`, subclass `RetargetingSolver`
2. Late-import heavy deps inside `setup()`, not at module top-level
3. Register: `registry.register_solver(MySolver)` at the bottom of the file
4. Add the module path to `_auto_register()` in `sensei/registry.py`
5. Add a test in `tests/test_solvers/`
6. Verify `pytest tests/test_base/test_abc_compliance.py` still passes

See [AGENTS.md](AGENTS.md) for the step-by-step agent workflow.

---

## Common pitfalls

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: general_motion_retargeting` | GMR not installed | `pip install -e third_party/GMR` |
| `ModuleNotFoundError: pinocchio` | Pinocchio not installed | `conda install -c conda-forge pinocchio casadi ipopt` |
| `import pinocchio.casadi` fails | pip-installed pin lacks CasADi bridge | Reinstall via conda-forge |
| `AssertionError: Call setup() before solve()` | Lifecycle violated | Always call `setup(robot)` first |
| `IPOPT: Infeasible_Problem_Detected` | Bad warm-start or limits too tight | Increase `max_iter`; check q_prev within limits |
| `ValueError: Solver 'daqp' is not available` | DAQP QP backend missing | `pip install qpsolvers[daqp]` |
