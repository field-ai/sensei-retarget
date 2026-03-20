"""
Registry for sources, solvers, and metrics.

Usage:
    from sensei.registry import registry

    # Register (done in each module's __init__.py or at import time)
    registry.register_source(GVHMRSource)
    registry.register_solver(GMRSolver)
    registry.register_metric(SolverTimingMetric)

    # Look up by name
    solver_cls = registry.get_solver("gmr")
    solver = solver_cls()

    # List available (skips classes whose deps are not installed)
    print(registry.available_solvers())
"""
from __future__ import annotations

import importlib
import warnings
from typing import Type

from sensei.base.source import MotionSource
from sensei.base.solver import RetargetingSolver
from sensei.base.metric import Metric


class Registry:
    def __init__(self) -> None:
        self._sources: dict[str, Type[MotionSource]] = {}
        self._solvers: dict[str, Type[RetargetingSolver]] = {}
        self._metrics: dict[str, Type[Metric]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_source(self, cls: Type[MotionSource]) -> Type[MotionSource]:
        self._sources[cls().name] = cls
        return cls

    def register_solver(self, cls: Type[RetargetingSolver]) -> Type[RetargetingSolver]:
        self._solvers[cls().name] = cls
        return cls

    def register_metric(self, cls: Type[Metric]) -> Type[Metric]:
        self._metrics[cls().name] = cls
        return cls

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_source(self, name: str) -> Type[MotionSource]:
        if name not in self._sources:
            raise KeyError(f"Unknown source '{name}'. Available: {list(self._sources)}")
        return self._sources[name]

    def get_solver(self, name: str) -> Type[RetargetingSolver]:
        if name not in self._solvers:
            raise KeyError(f"Unknown solver '{name}'. Available: {list(self._solvers)}")
        return self._solvers[name]

    def get_metric(self, name: str) -> Type[Metric]:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric '{name}'. Available: {list(self._metrics)}")
        return self._metrics[name]

    # ── Availability (skips classes with missing deps) ────────────────────────

    def available_solvers(self) -> list[str]:
        return list(self._solvers.keys())

    def available_sources(self) -> list[str]:
        return list(self._sources.keys())

    def available_metrics(self) -> list[str]:
        return list(self._metrics.keys())


# Module-level singleton
registry = Registry()


def _auto_register() -> None:
    """
    Attempt to import all known source/solver/metric modules so they
    self-register. Modules with missing deps are skipped with a warning.
    """
    _modules = [
        "sensei.sources.gvhmr",
        "sensei.solvers.gmr",
        "sensei.solvers.pinocchio_ipopt",
        "sensei.metrics.timing",
    ]
    for mod in _modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            warnings.warn(
                f"Could not import {mod} (missing dependency: {e}). "
                "Skipping auto-registration.",
                stacklevel=2,
            )


_auto_register()
