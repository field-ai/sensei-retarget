"""Abstract base class for all metrics."""
from abc import ABC, abstractmethod
from sensei.types import MotionSequence, RobotMotion, MetricResult


class Metric(ABC):
    """
    Computes a scalar evaluation score over a (source, result) pair.

    Metrics are stateless — compute() may be called multiple times with
    different inputs without calling any setup method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'solver_timing', 'ee_position_error'."""
        ...

    @property
    @abstractmethod
    def unit(self) -> str:
        """Physical unit string, e.g. 'fps', 'ms', 'mm', 'deg', 'rad/s'."""
        ...

    @abstractmethod
    def compute(
        self,
        source: MotionSequence | None,
        result: RobotMotion,
    ) -> MetricResult:
        """
        Compute the metric.

        Args:
            source: the input MotionSequence (may be None for metrics that
                    only inspect the solver output, e.g. timing).
            result: the RobotMotion produced by the solver.

        Returns:
            MetricResult with .value as the scalar summary.
        """
        ...
