# AGENTS.md — Agent Navigation Guide

This file tells AI agents how to navigate, extend, and test this repo autonomously.

---

## Read this first

1. **[docs/plan.md](docs/plan.md)** — overall architecture, phase roadmap, design decisions
2. **[docs/phase1.md](docs/phase1.md)** — Phase 1 (done): GMR baseline, benchmark results
3. **[docs/phase2.md](docs/phase2.md)** — Phase 2 (active): Pinocchio + CasADi + IPOPT, collision-aware NLP
4. **[CLAUDE.md](CLAUDE.md)** — environment setup, run commands, failure modes
5. **[sensei/types.py](sensei/types.py)** — all shared data types; understand these before touching anything else

---

## Repo map

```
sensei/base/            The contract layer — the most important files
  source.py             MotionSource ABC: load(path) → MotionSequence
  solver.py             RetargetingSolver ABC: setup(robot) + solve(motion) → RobotMotion
  metric.py             Metric ABC: compute(source, result) → MetricResult

sensei/types.py         MotionSequence, RobotConfig, RobotMotion, MetricResult
                        float64 numpy throughout. No solver deps.

sensei/sources/         One file per upstream data format
  gvhmr.py              GVHMR .pt → MotionSequence (delegates FK to GMR)

sensei/solvers/         One file per solver — each fully self-contained
  gmr.py                GMR + mink differential IK (Phase 1 baseline)
  pinocchio_ipopt.py    Pinocchio + CasADi + IPOPT, full 29-DoF (Phase 2)

sensei/metrics/         One file per metric
  timing.py             FPS + latency percentiles (reads frame_times_s from metadata)
  accuracy.py           EE position + orientation error vs. SMPL-X targets (Phase 2)

sensei/robots/
  g1.py                 get_g1_config() → RobotConfig (joint limits from g1_mocap_29dof.xml)

sensei/registry.py      Auto-discovery registry; auto-imports on module load

tests/
  conftest.py           Fixtures: tennis_pt_path, mock_motion_sequence, g1_config
  test_base/            ABC compliance tests — run without solver deps
  test_sources/         Integration tests for each source
  test_solvers/         Integration tests for each solver
  test_metrics/         Unit tests for each metric

docs/
  plan.md               Master plan
  phase1.md             Phase 1 detailed build guide (done — results included)
  phase2.md             Phase 2 design: full-robot NLP, collision constraints, GMR comparison

third_party/            Repos we import from — gitignored
  GMR/                  pip install -e third_party/GMR
  GVHMR/                pip install -e third_party/GVHMR
  alpaqa/               pip install -e third_party/alpaqa (Phase 3)

reference_repos/        Read-only reference — never imported, gitignored
  mink/ G1Pilot/ xr_teleoperate/ pinocchio-casadi-examples/ …
```

---

## How to add a new Source

1. Create `sensei/sources/<name>.py`
2. Subclass `MotionSource` from `sensei.base.source`
3. Implement `name`, `can_load(path)`, `load(path) → MotionSequence`
4. Import heavy deps inside `load()` (late import)
5. If the source runs FK, populate `MotionSequence.landmarks`
6. At module bottom: `registry.register_source(MySource)`
7. Add to `sensei/registry.py` `_auto_register()` list
8. Add test `tests/test_sources/test_<name>_source.py`
9. Run: `pytest tests/test_base/test_abc_compliance.py tests/test_sources/`

Template:
```python
from sensei.base.source import MotionSource
from sensei.types import MotionSequence
from sensei.registry import registry

class MySource(MotionSource):
    @property
    def name(self) -> str: return "my_source"

    def can_load(self, path: str) -> bool: ...

    def load(self, path: str) -> MotionSequence:
        import my_dep   # late import
        ...

registry.register_source(MySource)
```

---

## How to add a new Solver

1. Create `sensei/solvers/<name>.py`
2. Subclass `RetargetingSolver` from `sensei.base.solver`
3. Implement `name`, `setup(robot)`, `solve_frame(targets, q_prev)`
4. Import solver deps inside `setup()` (late import)
5. Store robot config as `self._robot` — the base class `solve()` needs it
6. `solve()` is optional to override; the default loop handles timing automatically
7. At module bottom: `registry.register_solver(MySolver)`
8. Add to `sensei/registry.py` `_auto_register()` list
9. Add test `tests/test_solvers/test_<name>_solver.py`
10. Run: `pytest tests/test_base/test_abc_compliance.py tests/test_solvers/`

Template:
```python
import numpy as np
from sensei.base.solver import RetargetingSolver
from sensei.types import MotionSequence, RobotConfig
from sensei.registry import registry

class MySolver(RetargetingSolver):
    @property
    def name(self) -> str: return "my_solver"

    def setup(self, robot: RobotConfig) -> None:
        import my_solver_dep   # late import
        self._robot = robot
        # build model...

    def solve_frame(self, targets: dict, q_prev: np.ndarray):
        q = ...   # float64, shape (DoF,)
        return q, True   # (q, converged)

registry.register_solver(MySolver)
```

---

## How to add a new Metric

1. Create `sensei/metrics/<name>.py`
2. Subclass `Metric` from `sensei.base.metric`
3. Implement `name`, `unit`, `compute(source, result) → MetricResult`
4. `source` may be `None` for metrics that only inspect `RobotMotion`
5. At module bottom: `registry.register_metric(MyMetric)`
6. Add to `sensei/registry.py` `_auto_register()` list
7. Add test `tests/test_metrics/test_<name>_metric.py`

---

## How to run a benchmark

```bash
# Single clip, single solver
python scripts/run_pipeline.py \
    --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
    --source gvhmr --solver gmr --robot g1 --output outputs/tennis_gmr.pkl

# All four test clips
for clip in tennis basketball_clip dance_clip 0_input_video; do
    python scripts/run_pipeline.py \
        --input /mnt/code/GVHMR/outputs/demo/${clip}/hmr4d_results.pt \
        --output outputs/${clip}_g1_gmr.pkl
done
```

Expected output (Phase 1 target):
- FPS ≥ 30 (GMR published: 60–70 FPS on high-end CPUs)
- Convergence rate ≥ 90%
- `dof_pos` shape `(N, 29)`, values within G1 joint limits

---

## Key invariants — never break these

1. **`sensei/base/` and `sensei/types.py`** — no solver-specific imports, ever
2. **All arrays are `float64` numpy** — convert at source load time, not inside solvers
3. **Late imports** — every solver imports its heavy deps inside `setup()`, not at top-level
4. **`self._robot`** — every solver must store the `RobotConfig` as `self._robot` for the base class `solve()` loop to work
5. **`test_abc_compliance.py` must pass with zero solver deps installed** — it tests class structure only
6. **`third_party/` and `reference_repos/` are never committed** — they are gitignored

---

## Data paths (hardcoded, always available)

```
GVHMR test clips:  /mnt/code/GVHMR/outputs/demo/{tennis,basketball_clip,dance_clip,0_input_video}/hmr4d_results.pt
SMPL-X body model: /mnt/code/GMR/assets/body_models/smplx/
G1 MuJoCo XML:     /mnt/code/GMR/assets/unitree_g1/g1_mocap_29dof.xml
G1 URDF (Phase 2): reference_repos/xr_teleoperate/assets/g1/g1_body29_hand14.urdf
```

---

## Testing rules

- `pytest tests/test_base/` — always runs; no deps beyond numpy+scipy
- `pytest tests/` — requires `sensei` conda env with GMR installed
- Do not mock solver internals in integration tests — test against real GVHMR clips
- Keep individual test clips to ≤ 30 frames to keep CI fast; use full clips in benchmarks
- After any change to `sensei/base/` or `sensei/types.py`, run the full test suite

---

## Current phase: Phase 1

See [docs/phase1.md](docs/phase1.md) for the detailed build guide.

Remaining Phase 1 tasks:
- [ ] `scripts/run_pipeline.py` — CLI entry point
- [ ] `tests/test_sources/test_gvhmr_source.py`
- [ ] `tests/test_solvers/test_gmr_solver.py`
- [ ] `tests/test_metrics/test_timing_metric.py`
- [ ] Verify end-to-end: four test clips produce valid G1 motion + timing report
