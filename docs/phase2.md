# Phase 2 — Pinocchio + CasADi + IPOPT Solver

> **Goal**: Replace GMR's QP-based IK with a full NLP formulated in Pinocchio + CasADi and solved by IPOPT. Phase 2a establishes the baseline NLP (kinematics only). Phase 2b adds ground and self-collision constraints inside the optimiser.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Data Flow](#data-flow)
4. [Phase 2a — Kinematic NLP](#phase-2a--kinematic-nlp)
   - [Joint Reduction](#joint-reduction)
   - [CasADi OCP Structure](#casadi-ocp-structure)
   - [Cost Terms](#cost-terms)
   - [Constraints](#constraints)
   - [IPOPT Options](#ipopt-options)
   - [Warm-Start Strategy](#warm-start-strategy)
5. [Phase 2b — Collision-Aware NLP](#phase-2b--collision-aware-nlp)
   - [Ground Collision](#ground-collision)
   - [Self-Collision](#self-collision)
   - [Design Decision: Symbolic vs Linearised Constraints](#design-decision-symbolic-vs-linearised-constraints)
6. [Implementation Plan](#implementation-plan)
   - [New Files](#new-files)
   - [File: sensei/solvers/pinocchio_ipopt.py](#file-senseisolverspinocchio_ipoptpy)
7. [Metrics Added in Phase 2](#metrics-added-in-phase-2)
8. [Acceptance Criteria](#acceptance-criteria)
9. [Reference Implementations](#reference-implementations)

---

## Overview

| | Phase 2a | Phase 2b |
|--|---------|---------|
| **Solver** | Pinocchio + CasADi + IPOPT | same + collision constraints |
| **Collision** | none | ground plane + self |
| **Key file** | `sensei/solvers/pinocchio_ipopt.py` | same file, toggled via flag |
| **New metric** | `AccuracyMetric` | (same) |
| **Benchmark vs** | Phase 1 (GMR/mink) | Phase 2a |

The two phases share one solver class (`PinocchioIPOPTSolver`). Collisions are activated by passing `collision=True` to the constructor — enabling apples-to-apples comparison without code duplication.

---

## Environment Setup

```bash
# Pinocchio + CasADi — conda-forge binary bundles the pinocchio.casadi symbolic sub-module.
# pip install pin works but the .casadi bridge is unreliable.
conda install -c conda-forge pinocchio casadi

# IPOPT + Python bindings
conda install -c conda-forge ipopt cyipopt

# Verify
python -c "import pinocchio as pin; print(pin.__version__)"
python -c "import pinocchio.casadi as cpin; print('cpin ok')"
python -c "import casadi; print(casadi.__version__)"
python -c "import cyipopt; print('cyipopt ok')"
```

---

## Data Flow

```
MotionSequence
    │  .landmarks[i]  {joint_name: (pos (3,), rot (Rotation))}
    │
    ▼ PinocchioIPOPTSolver.setup(robot)
    │  • pin.RobotWrapper.BuildFromURDF(robot.urdf_path)
    │  • buildReducedRobot() — lock legs + waist + hands
    │  • cpin.Model(reduced_model)  — CasADi symbolic copy
    │  • Build CasADi Functions once (FK, EE error, cost, grad)
    │  • Build casadi.Opti() problem — set up variables, parameters, cost, constraints
    │
    ▼ PinocchioIPOPTSolver.solve_frame(targets, q_prev)
    │  • Set Opti parameters: p_ref_L, R_ref_L, p_ref_R, R_ref_R, q_last
    │  • opti.set_initial(var_q, q_prev)   ← warm start
    │  • sol = opti.solve()
    │  • return sol.value(var_q), converged
    │
    ▼ RobotMotion  (same schema as Phase 1)
    │
    ▼ AccuracyMetric  (new in Phase 2)
       • FK(q) → EE positions/orientations
       • Compare against SMPL-X targets
       • Report: position error (mm), orientation error (deg)
```

---

## Phase 2a — Kinematic NLP

### Full Robot IK

We solve IK for all 29 DoF simultaneously — legs, waist, and arms together. This lets the optimiser trade off between upper- and lower-body joint angles to best satisfy all EE targets at once, and makes the ground-collision constraints in Phase 2b meaningful (they act on the actual leg configuration).

No `buildReducedRobot()` — the full Pinocchio model is used directly.

### CasADi OCP Structure

```python
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi

# --- Build symbolic model (once at setup()) ---
cmodel = cpin.Model(reduced.model)
cdata  = cmodel.createData()

nq = cmodel.nq

# CasADi function for FK (computed once, compiled to C via CasADi)
cq = casadi.SX.sym("q", nq)
cpin.framesForwardKinematics(cmodel, cdata, cq)

p_L = cdata.oMf[L_ee_id].translation        # (3,) SX
R_L = cdata.oMf[L_ee_id].rotation           # (3,3) SX
p_R = cdata.oMf[R_ee_id].translation
R_R = cdata.oMf[R_ee_id].rotation

fk_fn = casadi.Function("fk", [cq], [p_L, R_L, p_R, R_R])

# --- Opti problem (built once at setup(), called each frame) ---
opti = casadi.Opti()

var_q      = opti.variable(nq)           # optimisation variable
par_q_last = opti.parameter(nq)          # warm-start / smoothing reference
par_p_L    = opti.parameter(3)           # target left wrist position
par_R_L    = opti.parameter(3, 3)        # target left wrist rotation
par_p_R    = opti.parameter(3)
par_R_R    = opti.parameter(3, 3)
```

### Cost Terms

Following G1Pilot + xr_teleoperate with log-space rotation error:

```python
p_L_fk, R_L_fk, p_R_fk, R_R_fk = fk_fn(var_q)

# Translation error (metres)
err_trans_L = p_L_fk - par_p_L
err_trans_R = p_R_fk - par_p_R

# Rotation error in so(3) tangent space
err_rot_L = cpin.log3(R_L_fk @ par_R_L.T)   # (3,) SX
err_rot_R = cpin.log3(R_R_fk @ par_R_R.T)

# Cost
trans_cost = casadi.sumsqr(err_trans_L) + casadi.sumsqr(err_trans_R)
rot_cost   = casadi.sumsqr(err_rot_L)   + casadi.sumsqr(err_rot_R)
reg_cost   = casadi.sumsqr(var_q)
smooth_cost= casadi.sumsqr(var_q - par_q_last)

opti.minimize(
    50.0  * trans_cost   +   # (m²)  — dominant term
     3.0  * rot_cost     +   # (rad²)
     0.02 * reg_cost     +   # pull toward zero configuration
     0.1  * smooth_cost      # pull toward previous frame
)
```

Weights are taken from G1Pilot and xr_teleoperate; adjust once benchmarking accuracy.

### Constraints

```python
q_min = reduced.model.lowerPositionLimit
q_max = reduced.model.upperPositionLimit

opti.subject_to(opti.bounded(q_min, var_q, q_max))
```

No velocity constraints in 2a (unlike GMR/mink which solves in velocity space). Smoothness is achieved via `smooth_cost`.

### IPOPT Options

```python
opts = {
    "expand":                     True,     # inline CasADi graph → faster
    "detect_simple_bounds":       True,     # treat box constraints natively
    "ipopt.print_level":          0,
    "ipopt.max_iter":             50,
    "ipopt.tol":                  1e-4,
    "ipopt.acceptable_tol":       5e-4,
    "ipopt.acceptable_iter":      5,        # stop early if acceptable for 5 iters
    "ipopt.warm_start_init_point":"yes",
}
opti.solver("ipopt", opts)
```

`max_iter=50` is a deliberate real-time budget. `acceptable_tol` allows early exit once the solution is "good enough" — critical for latency.

### Warm-Start Strategy

```python
def solve_frame(self, targets, q_prev):
    # Parameters
    self._opti.set_value(self._par_q_last, q_prev)
    self._opti.set_value(self._par_p_L,    targets["left_wrist_pos"])
    self._opti.set_value(self._par_R_L,    targets["left_wrist_rot"])
    self._opti.set_value(self._par_p_R,    targets["right_wrist_pos"])
    self._opti.set_value(self._par_R_R,    targets["right_wrist_rot"])

    # Warm start from previous frame solution
    self._opti.set_initial(self._var_q, q_prev)

    try:
        sol = self._opti.solve()
        return sol.value(self._var_q).astype(np.float64), True
    except RuntimeError:
        # IPOPT failed to converge — return previous solution
        return q_prev.copy(), False
```

CasADi's `opti.solve()` reuses the previous primal/dual variables automatically when `warm_start_init_point=yes`. Explicitly setting `set_initial` additionally seeds the primal.

---

## Phase 2b — Collision-Aware NLP

### Ground Collision

**What we want**: no foot (or any foot-adjacent link) penetrates the floor plane at `z = 0`.

Since FK is already symbolic via `pinocchio.casadi`, foot positions are directly available as CasADi SX expressions — no approximation needed:

```python
# Add to setup() after building fk_fn
cpin.framesForwardKinematics(cmodel, cdata, cq)

left_foot_z  = cdata.oMf[left_foot_id].translation[2]   # SX scalar
right_foot_z = cdata.oMf[right_foot_id].translation[2]

# Add to Opti problem
GROUND_Z       = 0.0
FOOT_CLEARANCE = 0.02   # 2 cm minimum — avoids numerical grazing

opti.subject_to(left_foot_z  >= GROUND_Z + FOOT_CLEARANCE)
opti.subject_to(right_foot_z >= GROUND_Z + FOOT_CLEARANCE)
```

This is an exact symbolic constraint — IPOPT gets analytical gradients for free through CasADi's autodiff.

> The ground plane level is set from the root position in Phase 1. If we start tracking dynamic ground planes (sloped surfaces, stairs) in a later phase, `GROUND_Z` becomes a parameter.

### Self-Collision

**What we want**: key body part pairs stay separated by a minimum distance.

The challenge: Pinocchio's FCL distance queries (`pin.computeDistance`) are numerical, not symbolic — they can't be autodiff'd through CasADi. Two options:

**Option A — Sphere approximations (recommended for Phase 2b)**

Model each body part as a sphere with a fixed radius. The squared distance between two sphere centres is a symbolic CasADi expression (pure FK), and the constraint is exact for the sphere geometry:

```python
# Representative body-part spheres (centre = FK of named frame, radius hardcoded)
COLLISION_SPHERES = [
    # (frame_name,    radius_m)
    ("left_elbow_roll_link",  0.06),
    ("right_elbow_roll_link", 0.06),
    ("left_wrist_roll_link",  0.05),
    ("right_wrist_roll_link", 0.05),
    ("pelvis",                0.15),
    ("torso_link",            0.12),
]

# Pairs to check (skip kinematically adjacent links)
COLLISION_PAIRS = [
    ("left_wrist_roll_link",  "right_wrist_roll_link"),
    ("left_wrist_roll_link",  "pelvis"),
    ("right_wrist_roll_link", "pelvis"),
    ("left_elbow_roll_link",  "torso_link"),
    ("right_elbow_roll_link", "torso_link"),
]
```

```python
# Build symbolic distance functions (once at setup())
for frame_a, frame_b in COLLISION_PAIRS:
    p_a = cdata.oMf[frame_ids[frame_a]].translation   # (3,) SX
    p_b = cdata.oMf[frame_ids[frame_b]].translation
    r_a = SPHERE_RADII[frame_a]
    r_b = SPHERE_RADII[frame_b]

    # Signed distance between sphere surfaces
    d_sq     = casadi.sumsqr(p_a - p_b)
    d_min_sq = (r_a + r_b) ** 2

    # Constraint: ||p_a - p_b||² >= (r_a + r_b)²
    opti.subject_to(d_sq >= d_min_sq)
```

Squared-distance avoids a `sqrt()` in the symbolic graph (cheaper and numerically friendlier for IPOPT).

**Option B — Linearised FCL (future, Phase 3 or later)**

For mesh-accurate collision, compute FCL distance and the contact normal at the current linearisation point, then add a linearised half-space constraint:

```
d(q) ≈ d(q_k) + ∇_q d(q_k)ᵀ (q - q_k) ≥ d_min
```

This requires computing `∇_q d` via finite differences or by differentiating through Pinocchio's geometric Jacobians — more expensive and not needed for Phase 2b.

### Design Decision: Symbolic vs Linearised Constraints

| Property | Sphere approx (2b) | Linearised FCL (future) |
|----------|--------------------|------------------------|
| Differentiable through CasADi | ✅ exact | ⚠️ FD or manual |
| Geometry accuracy | sphere over-approximation | mesh-accurate |
| Setup cost | O(pairs) | O(pairs × FD_iters) per frame |
| IPOPT behaviour | constraint in original NLP | re-linearised each SQP iter |
| Implementation complexity | low | medium |

**Phase 2b uses sphere approximations.** They over-approximate collision bodies (conservative), which is safe and gives IPOPT exact gradients.

---

## Implementation Plan

### New Files

```
sensei/solvers/pinocchio_ipopt.py    # Phase 2a + 2b solver
sensei/metrics/accuracy.py           # AccuracyMetric (EE position + orientation error)
tests/test_solvers/test_pinocchio_ipopt.py
```

### File: `sensei/solvers/pinocchio_ipopt.py`

```python
class PinocchioIPOPTSolver(RetargetingSolver):
    """
    NLP-based retargeting solver: Pinocchio FK + CasADi symbolic diff + IPOPT.

    Phase 2a (collision=False): kinematic IK only.
    Phase 2b (collision=True):  + ground plane + sphere self-collision constraints.

    Args:
        collision:     enable ground + self-collision constraints (Phase 2b)
        max_iter:      IPOPT max iterations per frame (default 50)
        trans_weight:  cost weight for EE translation error (default 50.0)
        rot_weight:    cost weight for EE rotation error (default 3.0)
        reg_weight:    regularisation weight (default 0.02)
        smooth_weight: frame-to-frame smoothing weight (default 0.1)
    """

    def __init__(
        self,
        collision: bool = False,
        max_iter: int = 50,
        trans_weight: float = 50.0,
        rot_weight:   float = 3.0,
        reg_weight:   float = 0.02,
        smooth_weight:float = 0.1,
    ) -> None: ...

    def setup(self, robot: RobotConfig) -> None:
        # Late imports
        import pinocchio as pin
        import pinocchio.casadi as cpin
        import casadi

        # 1. Load full 29-DoF G1 URDF (no joint reduction)
        # 2. Build CasADi symbolic model (cpin.Model)
        # 3. Compile FK function for all EE frames
        # 4. Build Opti problem (variables, parameters, cost, constraints)
        # 5. If collision: add ground + self-collision constraints
        ...

    def solve_frame(self, targets: dict, q_prev: np.ndarray) -> tuple[np.ndarray, bool]:
        # Set parameters, warm start, solve, return (q, converged)
        ...

    @property
    def name(self) -> str:
        suffix = "_collision" if self._collision else ""
        return f"pinocchio_ipopt{suffix}"
```

### File: `sensei/metrics/accuracy.py`

```python
class AccuracyMetric(Metric):
    """
    End-effector accuracy vs. SMPL-X targets.

    Requires:
      - source.landmarks (from GVHMRSource) for EE targets
      - robot.urdf_path + reduced model for FK

    Reports:
      - position_error_mm:  mean Euclidean distance (mm) across EE × frames
      - rotation_error_deg: mean geodesic SO(3) distance (deg)
      - per_frame:          (N,) array of mean EE position error per frame
    """
```

---

## Metrics Added in Phase 2

| Metric | Unit | Notes |
|--------|------|-------|
| `AccuracyMetric` | mm, deg | EE position + orientation error vs. SMPL-X targets |
| `SolverTimingMetric` | ms | same as Phase 1, compare directly |

`scripts/plot_metrics.py` will be extended to show GMR vs 2a vs 2b side by side.

---

## Acceptance Criteria

### Phase 2a

- [ ] `pytest tests/test_solvers/test_pinocchio_ipopt.py` — passes
- [ ] `pytest tests/test_base/test_abc_compliance.py` — passes (GMR must still pass too)
- [ ] Runs on all three test clips without crashing
- [ ] Convergence rate ≥ 80% (IPOPT may fail harder poses)
- [ ] `AccuracyMetric`: EE position error computed and logged for all clips
- [ ] Timing benchmark updated: GMR vs 2a side-by-side in `outputs/metrics_timing.png`

### GMR vs 2a Comparison

`scripts/plot_metrics.py` is extended to run both solvers and produce a side-by-side panel:

| Metric | GMR (Phase 1) | 2a target |
|--------|--------------|-----------|
| Mean latency | 3–5 ms | < 100 ms (offline) |
| Convergence | 100% | ≥ 80% |
| EE position error | not tracked | baseline established |
| Joint violations | 0–4% | 0% (hard constraint) |

IPOPT is expected to be slower than GMR's QP — the goal of 2a is correctness and establishing accuracy numbers, not matching GMR's speed. Speed optimisation comes in Phase 3 (alpaqa).

### Phase 2b

- [ ] All Phase 2a criteria still pass
- [ ] Zero ground penetration events (left/right foot z ≥ 0) on all clips
- [ ] Zero active self-collision violations (sphere pairs) on all clips
- [ ] Mean solve latency < 100 ms (10 fps minimum — lower bound for offline use)
- [ ] Accuracy does not regress vs. 2a by more than 10 mm

---

## Reference Implementations

| What | Where | Key insight |
|------|-------|------------|
| Joint reduction | `reference_repos/xr_teleoperate/teleop/robot_control/robot_arm_ik.py` | `buildReducedRobot()`, which joints to lock, EE frame offsets |
| Cost + IPOPT options | `reference_repos/G1Pilot/src/arm_ik/robot_arm_ik.cpp` | Weights (50/3/0.02/0.1), `log6` SE3 error, `acceptable_iter=5` |
| CasADi Opti pattern | `reference_repos/pinocchio-casadi-examples/utils/ocp_utils.py` | `casadi.Function` compilation, `framePlacementCost` helper |
| Friction cone / ground | `reference_repos/pinocchio-casadi-examples/solo_geom_inv.py` | foot height constraint pattern |
| Collision filtering | `reference_repos/G1Pilot/include/arm_ik/robot_arm_ik.h` | `filter_adjacent_collision_pairs()` |
| Full-body NLP | `reference_repos/pinocchio-casadi-examples/talos_yoga.py` | multi-contact OCP structure |
