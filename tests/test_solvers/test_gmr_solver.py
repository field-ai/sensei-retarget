"""
Integration tests for GMRSolver.
Requires: pip install -e third_party/GMR
"""
import numpy as np
import pytest
from sensei.solvers.gmr import GMRSolver
from sensei.sources.gvhmr import GVHMRSource
from sensei.metrics.timing import SolverTimingMetric

pytestmark = pytest.mark.integration

_SHORT = 30   # frames to use in fast tests (≈1 second of motion)


@pytest.fixture(scope="module")
def tennis_motion_short(tennis_pt_path):
    motion = GVHMRSource().load(tennis_pt_path)
    # Slice to first _SHORT frames
    motion.num_frames = _SHORT
    motion.body_pose   = motion.body_pose[:_SHORT]
    motion.global_orient = motion.global_orient[:_SHORT]
    motion.transl      = motion.transl[:_SHORT]
    motion.landmarks   = motion.landmarks[:_SHORT]
    return motion


@pytest.fixture(scope="module")
def gmr_result(tennis_motion_short, g1_config):
    solver = GMRSolver(verbose=False)
    solver.setup(g1_config)
    result = solver.solve(tennis_motion_short)
    solver.teardown()
    return result


# ── Output structure ──────────────────────────────────────────────────────────

def test_result_frame_count(gmr_result):
    assert gmr_result.num_frames == _SHORT

def test_result_robot_and_solver_name(gmr_result):
    assert gmr_result.robot_name == "unitree_g1"
    assert gmr_result.solver_name == "gmr"

def test_q_shape_and_dtype(gmr_result, g1_config):
    q_arr = gmr_result.q_array()
    assert q_arr.shape == (_SHORT, g1_config.num_dof)
    assert q_arr.dtype == np.float64

def test_dq_shape(gmr_result, g1_config):
    dq_arr = gmr_result.dq_array()
    assert dq_arr.shape == (_SHORT, g1_config.num_dof)

def test_timestamps_monotonic(gmr_result):
    ts = [s.timestamp for s in gmr_result.states]
    assert all(ts[i] < ts[i+1] for i in range(len(ts) - 1))

def test_frame_times_recorded(gmr_result):
    assert "frame_times_s" in gmr_result.metadata
    ft = gmr_result.metadata["frame_times_s"]
    assert len(ft) == _SHORT
    assert np.all(ft > 0)


# ── Quality checks ────────────────────────────────────────────────────────────

def test_convergence_rate(gmr_result):
    rate = np.mean(gmr_result.converged)
    assert rate >= 0.90, f"Convergence rate too low: {rate:.1%}"

def test_joint_limits_mostly_respected(gmr_result, g1_config):
    q = gmr_result.q_array()
    violations = np.mean(
        np.any((q < g1_config.joint_lower) | (q > g1_config.joint_upper), axis=1)
    )
    assert violations <= 0.10, f"Joint limit violations: {violations:.1%}"

def test_q_not_all_zeros(gmr_result):
    q = gmr_result.q_array()
    assert not np.allclose(q, 0.0), "All joint angles are zero — solver likely failed"


# ── Timing ────────────────────────────────────────────────────────────────────

def test_fps_above_minimum(gmr_result):
    mr = SolverTimingMetric().compute(None, gmr_result)
    assert mr.value >= 30.0, f"FPS too low: {mr.value:.1f} (target ≥ 30)"


# ── solve_frame ───────────────────────────────────────────────────────────────

def test_solve_frame_returns_correct_shape(tennis_motion_short, g1_config):
    solver = GMRSolver(verbose=False)
    solver.setup(g1_config)
    targets = tennis_motion_short.landmarks[0]
    q_prev = g1_config.default_pose.copy()
    q, converged = solver.solve_frame(targets, q_prev)
    solver.teardown()

    assert q.shape == (g1_config.num_dof,)
    assert q.dtype == np.float64
    assert isinstance(converged, bool)


# ── setup must be called before solve ────────────────────────────────────────

def test_solve_without_setup_raises(tennis_motion_short):
    solver = GMRSolver()
    with pytest.raises(AssertionError):
        solver.solve(tennis_motion_short)


# ── Missing landmarks raises ──────────────────────────────────────────────────

def test_solve_without_landmarks_raises(g1_config, mock_motion_sequence):
    solver = GMRSolver()
    solver.setup(g1_config)
    with pytest.raises(AssertionError, match="landmarks"):
        solver.solve(mock_motion_sequence)
    solver.teardown()
