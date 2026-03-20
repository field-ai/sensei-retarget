"""Abstract base class for all motion sources."""
from abc import ABC, abstractmethod
from sensei.types import MotionSequence


class MotionSource(ABC):
    """
    Produces a MotionSequence from an upstream data file.

    Implementations must do any format-specific parsing, unit conversion,
    and (if needed) SMPL-X FK to populate MotionSequence.landmarks.
    No solver-specific logic belongs here.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in configs, logs, and CLI flags."""
        ...

    @abstractmethod
    def load(self, path: str) -> MotionSequence:
        """
        Load and preprocess the file at `path`.

        Must return a MotionSequence with float64 numpy arrays.
        Should populate .landmarks if this source runs FK.
        """
        ...

    @abstractmethod
    def can_load(self, path: str) -> bool:
        """Return True if this source can handle the given path."""
        ...
