"""
SolverTimingMetric: wall-clock FPS and per-frame latency statistics.

Reads frame_times_s from RobotMotion.metadata (recorded by the default
RetargetingSolver.solve() loop). No re-running of the solver.
"""
from __future__ import annotations

import numpy as np

from sensei.base.metric import Metric
from sensei.types import MotionSequence, RobotMotion, MetricResult


class SolverTimingMetric(Metric):
    """
    Computes solver FPS and latency percentiles from per-frame timing data.

    Requires RobotMotion.metadata['frame_times_s'] — a (N,) float64 array
    of wall-clock seconds per frame. This is written automatically by the
    default RetargetingSolver.solve() loop.

    .value  = mean FPS
    .unit   = "fps"
    .per_frame = per-frame latency in ms
    .metadata = {latency_p50_ms, latency_p95_ms, latency_p99_ms, ...}
    """

    @property
    def name(self) -> str:
        return "solver_timing"

    @property
    def unit(self) -> str:
        return "fps"

    def compute(
        self,
        source: MotionSequence | None,
        result: RobotMotion,
    ) -> MetricResult:
        frame_times_s = result.metadata.get("frame_times_s")
        if frame_times_s is None:
            raise ValueError(
                "SolverTimingMetric requires RobotMotion.metadata['frame_times_s']. "
                "Ensure the solver records per-frame timing (the default "
                "RetargetingSolver.solve() loop does this automatically)."
            )

        t = np.asarray(frame_times_s, dtype=np.float64)
        t_ms = t * 1000.0
        fps = 1.0 / float(np.mean(t)) if np.mean(t) > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=fps,
            unit=self.unit,
            per_frame=t_ms,
            metadata={
                "fps": fps,
                "latency_mean_ms": float(np.mean(t_ms)),
                "latency_p50_ms":  float(np.percentile(t_ms, 50)),
                "latency_p95_ms":  float(np.percentile(t_ms, 95)),
                "latency_p99_ms":  float(np.percentile(t_ms, 99)),
                "latency_min_ms":  float(np.min(t_ms)),
                "latency_max_ms":  float(np.max(t_ms)),
                "num_frames":      len(t_ms),
                "converge_rate":   float(np.mean(result.converged)),
            },
        )
