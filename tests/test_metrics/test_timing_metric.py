"""Unit tests for SolverTimingMetric — no solver deps required."""
import numpy as np
import pytest
from sensei.metrics.timing import SolverTimingMetric
from sensei.types import RobotMotion, RobotState


def _make_result(frame_times_s: np.ndarray, converged: np.ndarray | None = None) -> RobotMotion:
    N = len(frame_times_s)
    states = [RobotState(q=np.zeros(29), dq=np.zeros(29), timestamp=i / 30.0) for i in range(N)]
    return RobotMotion(
        robot_name="unitree_g1",
        solver_name="gmr",
        fps=30.0,
        states=states,
        converged=np.ones(N, dtype=bool) if converged is None else converged,
        metadata={"frame_times_s": frame_times_s},
    )


def test_fps_calculation():
    # 10 ms per frame → 100 FPS
    t = np.full(100, 0.010)
    mr = SolverTimingMetric().compute(None, _make_result(t))
    assert mr.value == pytest.approx(100.0, rel=0.01)
    assert mr.unit == "fps"


def test_latency_percentiles():
    rng = np.random.default_rng(42)
    t = rng.uniform(0.010, 0.030, 200)  # 10–30 ms
    mr = SolverTimingMetric().compute(None, _make_result(t))
    assert mr.metadata["latency_p50_ms"] < mr.metadata["latency_p95_ms"]
    assert mr.metadata["latency_p95_ms"] < mr.metadata["latency_p99_ms"]
    assert mr.metadata["latency_min_ms"] > 0


def test_per_frame_in_ms():
    t = np.full(50, 0.016)  # 16 ms
    mr = SolverTimingMetric().compute(None, _make_result(t))
    assert mr.per_frame is not None
    assert mr.per_frame.shape == (50,)
    np.testing.assert_allclose(mr.per_frame, 16.0, rtol=1e-6)


def test_converge_rate():
    t = np.full(10, 0.010)
    converged = np.array([True] * 8 + [False] * 2, dtype=bool)
    mr = SolverTimingMetric().compute(None, _make_result(t, converged))
    assert mr.metadata["converge_rate"] == pytest.approx(0.8)


def test_missing_frame_times_raises():
    result = _make_result(np.full(10, 0.010))
    del result.metadata["frame_times_s"]
    with pytest.raises(ValueError, match="frame_times_s"):
        SolverTimingMetric().compute(None, result)
