#!/usr/bin/env python3
"""
Side-by-side timing comparison: GMR vs PinocchioIPOPT on all GVHMR test clips.

Output: outputs/metrics_comparison.png

Usage:
    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --clips tennis basketball_clip
    python scripts/plot_metrics.py --solvers gmr          # GMR only
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

CLIPS = ["basketball_clip", "dance_clip", "0_input_video"]
CLIP_LABELS = {
    "basketball_clip": "Basketball",
    "dance_clip":      "Dance",
    "0_input_video":   "Tennis",
}
GVHMR_ROOT = pathlib.Path("/mnt/code/GVHMR/outputs/demo")

# Solver display names and colours
SOLVER_META = {
    "gmr":              {"label": "GMR (mink QP)",        "color": "#5B9BD5"},
    "pinocchio_ipopt":  {"label": "Pinocchio + IPOPT",    "color": "#ED7D31"},
}

# Plot style
plt.rcParams.update({
    "figure.facecolor":  "#0f0f0f",
    "axes.facecolor":    "#1a1a1a",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#ccc",
    "axes.titlecolor":   "#eee",
    "xtick.color":       "#999",
    "ytick.color":       "#999",
    "text.color":        "#ddd",
    "grid.color":        "#2a2a2a",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.8,
    "legend.facecolor":  "#1a1a1a",
    "legend.edgecolor":  "#444",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
})


# ── Per-solver runner ─────────────────────────────────────────────────────────

def _build_solver(name: str):
    if name == "gmr":
        from sensei.solvers.gmr import GMRSolver
        return GMRSolver(verbose=False)
    if name == "pinocchio_ipopt":
        from sensei.solvers.pinocchio_ipopt import PinocchioIPOPTSolver
        return PinocchioIPOPTSolver()
    raise ValueError(f"Unknown solver: {name}")


def run_clip(clip: str, solver_name: str) -> dict:
    """Run one solver on one clip; return timing + quality data."""
    from sensei.sources.gvhmr import GVHMRSource
    from sensei.robots.g1 import get_g1_config
    from sensei.metrics.timing import SolverTimingMetric

    pt = GVHMR_ROOT / clip / "hmr4d_results.pt"
    motion = GVHMRSource().load(str(pt))

    robot  = get_g1_config()
    solver = _build_solver(solver_name)
    solver.setup(robot)

    t0 = time.perf_counter()
    result = solver.solve(motion)
    wall_s = time.perf_counter() - t0
    solver.teardown()

    timing = SolverTimingMetric().compute(motion, result)

    q_arr = result.q_array()
    violations = np.mean(
        np.any((q_arr < robot.joint_lower) | (q_arr > robot.joint_upper), axis=1)
    )

    return {
        "clip":          clip,
        "solver":        solver_name,
        "label":         CLIP_LABELS[clip],
        "num_frames":    result.num_frames,
        "fps":           timing.metadata["fps"],
        "latency_ms":    timing.per_frame,           # (N,) ms
        "mean_ms":       timing.metadata["latency_mean_ms"],
        "p50":           timing.metadata["latency_p50_ms"],
        "p95":           timing.metadata["latency_p95_ms"],
        "p99":           timing.metadata["latency_p99_ms"],
        "converge_rate": timing.metadata["converge_rate"],
        "violations":    float(violations),
        "wall_s":        wall_s,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(data: dict[str, list[dict]], out_path: pathlib.Path) -> None:
    """
    data: {solver_name: [result_per_clip, ...]}

    Layout (3 rows):
      Row 0  — summary bars: mean latency (left) + FPS (right), grouped by clip
      Row 1  — per-clip time series: one subplot per clip, one line per solver
      Row 2  — per-clip latency histogram: one subplot per clip, overlaid
    """
    solvers = list(data.keys())
    clips   = [r["clip"]  for r in data[solvers[0]]]
    labels  = [r["label"] for r in data[solvers[0]]]
    nc      = len(clips)

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        "Solver Comparison — GMR vs Pinocchio+IPOPT  (Unitree G1, 29 DoF)",
        fontsize=14, fontweight="bold", color="#eee", y=0.985,
    )

    gs = gridspec.GridSpec(
        3, nc,
        figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.945, bottom=0.065,
        left=0.065, right=0.975,
        height_ratios=[1.1, 1.0, 1.0],
    )

    # ── Row 0: Summary bar charts (span full width, split into two halves) ────
    ax_lat = fig.add_subplot(gs[0, : nc//2 + nc%2])
    ax_fps = fig.add_subplot(gs[0, nc//2 + nc%2 :] if nc > 1 else gs[0, :])

    # If nc is odd / nc=3 → left half = 2 cols, right half = 1 col.
    # Better: always do a clean split as 2 axes spanning half each.
    # Re-add with explicit colspan:
    ax_lat.remove()
    ax_fps.remove()
    ax_lat = fig.add_subplot(gs[0, 0])
    ax_fps = fig.add_subplot(gs[0, 1])
    # Third column (if exists) becomes a summary table
    if nc >= 3:
        ax_tbl = fig.add_subplot(gs[0, 2])
        ax_tbl.axis("off")
    else:
        ax_tbl = None

    x   = np.arange(nc)
    nsv = len(solvers)
    w   = 0.7 / nsv  # bar width per solver

    for si, sv in enumerate(solvers):
        meta   = SOLVER_META[sv]
        col    = meta["color"]
        lbl    = meta["label"]
        offset = (si - (nsv-1)/2) * w

        mean_ms = [data[sv][ci]["mean_ms"] for ci in range(nc)]
        fps_v   = [data[sv][ci]["fps"]     for ci in range(nc)]

        bars = ax_lat.bar(x + offset, mean_ms, w, color=col, alpha=0.88,
                          label=lbl, zorder=3)
        for bar, v in zip(bars, mean_ms):
            ax_lat.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.15,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color="#ddd")

        bars2 = ax_fps.bar(x + offset, fps_v, w, color=col, alpha=0.88,
                           label=lbl, zorder=3)
        for bar, v in zip(bars2, fps_v):
            ax_fps.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color="#ddd")

    ax_lat.axhline(1000/30, color="#ff4444", lw=1.2, ls="--", label="33.3 ms")
    ax_lat.set_title("Mean Solve Latency")
    ax_lat.set_ylabel("ms / frame")
    ax_lat.set_xticks(x); ax_lat.set_xticklabels(labels)
    ax_lat.legend(fontsize=7.5, loc="upper right")
    ax_lat.grid(axis="y", zorder=0)
    ax_lat.set_ylim(0, max(
        data[sv][ci]["mean_ms"] for sv in solvers for ci in range(nc)
    ) * 1.45)

    ax_fps.set_title("Effective FPS (solver only)")
    ax_fps.set_ylabel("frames / second")
    ax_fps.set_xticks(x); ax_fps.set_xticklabels(labels)
    ax_fps.legend(fontsize=7.5, loc="upper right")
    ax_fps.grid(axis="y", zorder=0)
    ax_fps.set_ylim(0, max(
        data[sv][ci]["fps"] for sv in solvers for ci in range(nc)
    ) * 1.35)

    # Summary table (third column)
    if ax_tbl is not None:
        rows, cols_hdr = [], ["Solver", "Clip", "p50", "p95", "Convg", "Viol"]
        for sv in solvers:
            for ci in range(nc):
                r = data[sv][ci]
                rows.append([
                    SOLVER_META[sv]["label"][:18],
                    r["label"],
                    f"{r['p50']:.1f}",
                    f"{r['p95']:.1f}",
                    f"{r['converge_rate']*100:.0f}%",
                    f"{r['violations']*100:.1f}%",
                ])
        tbl = ax_tbl.table(
            cellText=rows, colLabels=cols_hdr,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.05, 1.35)
        # Style header
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor("#333")
            if ri == 0:
                cell.set_facecolor("#2a2a3a")
                cell.set_text_props(color="#eee", fontweight="bold")
            else:
                sv_idx = (ri - 1) // nc
                sv_name = solvers[sv_idx] if sv_idx < len(solvers) else solvers[-1]
                cell.set_facecolor("#1a1a1a")
                cell.set_text_props(color=SOLVER_META[sv_name]["color"])
        ax_tbl.set_title("Summary Table", pad=8)

    # ── Row 1: Per-clip time series ───────────────────────────────────────────
    for ci, (clip, clabel) in enumerate(zip(clips, labels)):
        ax = fig.add_subplot(gs[1, ci])
        for sv in solvers:
            r   = data[sv][ci]
            col = SOLVER_META[sv]["color"]
            t   = r["latency_ms"]
            frames = np.arange(len(t))
            ax.fill_between(frames, t, alpha=0.12, color=col)
            ax.plot(frames, t, lw=0.9, color=col,
                    label=f"{SOLVER_META[sv]['label']} p50={r['p50']:.0f}ms")
            ax.axhline(r["p50"], color=col, lw=0.8, ls="--", alpha=0.55)

        ax.set_title(clabel)
        ax.set_xlabel("frame")
        if ci == 0:
            ax.set_ylabel("latency (ms)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(zorder=0)
        ax.set_xlim(0, max(len(data[sv][ci]["latency_ms"]) for sv in solvers) - 1)

    # ── Row 2: Latency distribution (overlaid histograms) ────────────────────
    for ci, (clip, clabel) in enumerate(zip(clips, labels)):
        ax = fig.add_subplot(gs[2, ci])
        all_ms = [v for sv in solvers for v in data[sv][ci]["latency_ms"]]
        bins = np.linspace(0, np.percentile(all_ms, 99) * 1.1, 35)

        for sv in solvers:
            r   = data[sv][ci]
            col = SOLVER_META[sv]["color"]
            ax.hist(r["latency_ms"], bins=bins, color=col, alpha=0.55,
                    edgecolor="#0f0f0f", lw=0.3, zorder=3,
                    label=SOLVER_META[sv]["label"])
            ax.axvline(r["p50"], color=col, lw=1.0, ls="--", alpha=0.8)
            ax.axvline(r["p95"], color=col, lw=1.0, ls=":",  alpha=0.7)

        ax.set_title(f"{clabel} — distribution")
        ax.set_xlabel("latency (ms)")
        if ci == 0:
            ax.set_ylabel("frame count")
        ax.legend(fontsize=7)
        ax.grid(axis="y", zorder=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot_metrics] saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side solver timing comparison."
    )
    parser.add_argument("--clips",   nargs="+", default=CLIPS)
    parser.add_argument("--solvers", nargs="+", default=list(SOLVER_META.keys()))
    parser.add_argument("--output",  default="outputs/metrics_comparison.png")
    args = parser.parse_args()

    # Collect: data[solver][clip_index]
    data: dict[str, list[dict]] = {}
    for sv in args.solvers:
        data[sv] = []
        for clip in args.clips:
            print(f"[plot_metrics] {sv} / {clip} …")
            t0 = time.perf_counter()
            r  = run_clip(clip, sv)
            elapsed = time.perf_counter() - t0
            data[sv].append(r)
            print(f"  {r['num_frames']} frames  |  {r['fps']:.0f} fps  "
                  f"p50={r['p50']:.1f}ms  p95={r['p95']:.1f}ms  "
                  f"convg={r['converge_rate']*100:.0f}%  "
                  f"viol={r['violations']*100:.1f}%  "
                  f"(wall {r['wall_s']:.1f}s)")

    print("[plot_metrics] generating figure …")
    plot(data, pathlib.Path(args.output))


if __name__ == "__main__":
    main()
