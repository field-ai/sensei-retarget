"""
Verify that all registered sources, solvers, and metrics:
  1. Are proper subclasses of their ABC
  2. Have a non-empty .name property
  3. Have the required abstract methods implemented

These tests run without any heavy deps (mink, mujoco, etc.) because they only
check class structure, not execution.
"""
import pytest
from sensei.base import MotionSource, RetargetingSolver, Metric


# ── Sources ───────────────────────────────────────────────────────────────────

class TestGVHMRSource:
    def test_is_subclass(self):
        from sensei.sources.gvhmr import GVHMRSource
        assert issubclass(GVHMRSource, MotionSource)

    def test_has_name(self):
        from sensei.sources.gvhmr import GVHMRSource
        assert GVHMRSource().name == "gvhmr"

    def test_has_required_methods(self):
        from sensei.sources.gvhmr import GVHMRSource
        src = GVHMRSource()
        assert callable(src.load)
        assert callable(src.can_load)

    def test_can_load_rejects_wrong_extension(self):
        from sensei.sources.gvhmr import GVHMRSource
        src = GVHMRSource()
        assert not src.can_load("motion.pkl")
        assert not src.can_load("motion.mp4")
        assert not src.can_load("nonexistent.pt")


# ── Solvers ───────────────────────────────────────────────────────────────────

class TestGMRSolver:
    def test_is_subclass(self):
        from sensei.solvers.gmr import GMRSolver
        assert issubclass(GMRSolver, RetargetingSolver)

    def test_has_name(self):
        from sensei.solvers.gmr import GMRSolver
        assert GMRSolver().name == "gmr"

    def test_has_required_methods(self):
        from sensei.solvers.gmr import GMRSolver
        s = GMRSolver()
        assert callable(s.setup)
        assert callable(s.solve)
        assert callable(s.solve_frame)
        assert callable(s.teardown)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestSolverTimingMetric:
    def test_is_subclass(self):
        from sensei.metrics.timing import SolverTimingMetric
        assert issubclass(SolverTimingMetric, Metric)

    def test_has_name_and_unit(self):
        from sensei.metrics.timing import SolverTimingMetric
        m = SolverTimingMetric()
        assert m.name == "solver_timing"
        assert m.unit == "fps"

    def test_has_compute(self):
        from sensei.metrics.timing import SolverTimingMetric
        assert callable(SolverTimingMetric().compute)


# ── Types ─────────────────────────────────────────────────────────────────────

def test_motion_sequence_dtypes(mock_motion_sequence):
    ms = mock_motion_sequence
    import numpy as np
    assert ms.body_pose.dtype == np.float64
    assert ms.global_orient.dtype == np.float64
    assert ms.transl.dtype == np.float64
    assert ms.betas.dtype == np.float64


def test_robot_config_shapes(g1_config):
    import numpy as np
    cfg = g1_config
    assert len(cfg.joint_names) == 29
    assert cfg.joint_lower.shape == (29,)
    assert cfg.joint_upper.shape == (29,)
    assert cfg.vel_limits.shape == (29,)
    assert cfg.default_pose.shape == (29,)
    assert np.all(cfg.joint_lower < cfg.joint_upper)
