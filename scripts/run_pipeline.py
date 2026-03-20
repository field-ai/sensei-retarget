#!/usr/bin/env python3
"""
Run a source + solver on a single input file and print a timing report.

Usage:
    python scripts/run_pipeline.py \\
        --input /mnt/code/GVHMR/outputs/demo/tennis/hmr4d_results.pt \\
        --source gvhmr --solver gmr --robot g1 \\
        --output outputs/tennis_g1_gmr.pkl

    # Run all four test clips
    for clip in tennis basketball_clip dance_clip 0_input_video; do
        python scripts/run_pipeline.py \\
            --input /mnt/code/GVHMR/outputs/demo/${clip}/hmr4d_results.pt \\
            --output outputs/${clip}_g1_gmr.pkl
    done
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np

# Ensure repo root is on path when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def build_source(name: str):
    if name == "gvhmr":
        from sensei.sources.gvhmr import GVHMRSource
        return GVHMRSource()
    raise ValueError(f"Unknown source '{name}'. Available: gvhmr")


def build_solver(name: str):
    if name == "gmr":
        from sensei.solvers.gmr import GMRSolver
        return GMRSolver(verbose=False)
    if name in ("pinocchio_ipopt", "pinocchio_ipopt_collision"):
        from sensei.solvers.pinocchio_ipopt import PinocchioIPOPTSolver
        collision = name.endswith("_collision")
        return PinocchioIPOPTSolver(collision=collision)
    raise ValueError(f"Unknown solver '{name}'. Available: gmr, pinocchio_ipopt, pinocchio_ipopt_collision")


def build_robot(name: str):
    if name == "g1":
        from sensei.robots.g1 import get_g1_config
        return get_g1_config()
    raise ValueError(f"Unknown robot '{name}'. Available: g1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sensei retargeting pipeline.")
    parser.add_argument("--input",  required=True,  help="Path to input motion file")
    parser.add_argument("--source", default="gvhmr", help="Source name (default: gvhmr)")
    parser.add_argument("--solver", default="gmr",   help="Solver name (default: gmr)")
    parser.add_argument("--robot",  default="g1",    help="Robot name (default: g1)")
    parser.add_argument("--output", default=None,    help="Save robot motion to .pkl")
    args = parser.parse_args()

    # ── Build components ──────────────────────────────────────────────────────
    source = build_source(args.source)
    solver = build_solver(args.solver)
    robot  = build_robot(args.robot)

    if not source.can_load(args.input):
        print(f"[error] Source '{args.source}' cannot load: {args.input}")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[sensei] loading  {args.input}")
    t0 = time.perf_counter()
    motion = source.load(args.input)
    load_s = time.perf_counter() - t0
    print(f"[sensei] loaded   {motion.num_frames} frames @ {motion.fps:.0f} fps  ({load_s:.2f}s)")

    # ── Setup + solve ─────────────────────────────────────────────────────────
    print(f"[sensei] solver   {solver.name}  →  robot {robot.name} ({robot.num_dof} DoF)")
    solver.setup(robot)
    try:
        t0 = time.perf_counter()
        result = solver.solve(motion)
        solve_s = time.perf_counter() - t0
    finally:
        solver.teardown()

    # ── Metrics ───────────────────────────────────────────────────────────────
    from sensei.metrics.timing import SolverTimingMetric
    timing = SolverTimingMetric().compute(motion, result)

    print(f"\n{'─'*50}")
    print(f"  frames        : {result.num_frames}")
    print(f"  wall time     : {solve_s:.2f}s")
    print(f"  FPS           : {timing.value:.1f}")
    print(f"  latency p50   : {timing.metadata['latency_p50_ms']:.1f} ms")
    print(f"  latency p95   : {timing.metadata['latency_p95_ms']:.1f} ms")
    print(f"  latency p99   : {timing.metadata['latency_p99_ms']:.1f} ms")
    print(f"  converge rate : {timing.metadata['converge_rate']*100:.1f}%")
    print(f"{'─'*50}\n")

    # Joint limit check
    q_arr = result.q_array()
    violations = np.mean(
        np.any((q_arr < robot.joint_lower) | (q_arr > robot.joint_upper), axis=1)
    )
    print(f"  joint limit violations: {violations*100:.1f}% of frames")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        N = result.num_frames
        data = {
            "fps":      result.fps,
            "root_pos": result.metadata.get("root_pos", np.zeros((N, 3), dtype=np.float64)),
            "root_rot": result.metadata.get("root_rot", np.tile([1.,0.,0.,0.], (N,1)).astype(np.float64)),
            "dof_pos":  result.q_array(),
            "solver":   result.solver_name,
            "robot":    result.robot_name,
        }
        with open(out, "wb") as f:
            pickle.dump(data, f)
        print(f"[sensei] saved → {out}")


if __name__ == "__main__":
    main()
