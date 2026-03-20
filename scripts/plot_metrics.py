#!/usr/bin/env python3
"""
Run SolverTimingMetric on all GVHMR test clips and produce a summary figure.

Output: outputs/metrics_timing.png

Usage:
    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --clips tennis basketball_clip
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
    "tennis":          "Tennis",
    "basketball_clip": "Basketball",
    "dance_clip":      "Dance",
    "0_input_video":   "Tennis",
}
GVHMR_ROOT = pathlib.Path("/mnt/code/GVHMR/outputs/demo")

# Palette — one colour per clip
_PALETTE = ["#5B9BD5", "#ED7D31", "#70AD47", "#9E6BB5"]

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


def run_clip(clip: str) -> dict:
    """Run GMR solver on one clip, return timing data (GMR solve only)."""
    from sensei.sources.gvhmr import GVHMRSource
    from sensei.solvers.gmr import GMRSolver
    from sensei.robots.g1 import get_g1_config
    from sensei.metrics.timing import SolverTimingMetric

    pt = GVHMR_ROOT / clip / "hmr4d_results.pt"

    # Load motion — not profiled (GVHMR pre-processing, not GMR)
    motion = GVHMRSource().load(str(pt))

    # GMR solve — this is what we profile
    robot  = get_g1_config()
    solver = GMRSolver(verbose=False)
    solver.setup(robot)
    t0 = time.perf_counter()
    result = solver.solve(motion)
    gmr_wall_s = time.perf_counter() - t0
    solver.teardown()

    # SolverTimingMetric uses frame_times_s recorded inside solve() — GMR only
    timing = SolverTimingMetric().compute(motion, result)

    q_arr = result.q_array()
    violations_per_frame = np.any(
        (q_arr < robot.joint_lower) | (q_arr > robot.joint_upper), axis=1
    ).astype(float)

    return {
        "clip":         clip,
        "label":        CLIP_LABELS[clip],
        "num_frames":   result.num_frames,
        "fps":          timing.metadata["fps"],
        "latency_ms":   timing.per_frame,          # (N,) ms per frame
        "p50":          timing.metadata["latency_p50_ms"],
        "p95":          timing.metadata["latency_p95_ms"],
        "p99":          timing.metadata["latency_p99_ms"],
        "mean_ms":      timing.metadata["latency_mean_ms"],
        "converge_rate": timing.metadata["converge_rate"],
        "violations":   violations_per_frame,
        "gmr_wall_s":   gmr_wall_s,
    }


def plot(results: list[dict], out_path: pathlib.Path) -> None:
    n = len(results)
    labels = [r["label"] for r in results]
    colors = _PALETTE[:n]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("GMR Solver — Timing Metrics  (Unitree G1, 29 DoF)",
                 fontsize=14, fontweight="bold", color="#eee", y=0.98)

    gs = gridspec.GridSpec(
        3, n,
        figure=fig,
        hspace=0.52,
        wspace=0.35,
        top=0.93, bottom=0.07, left=0.07, right=0.97,
        height_ratios=[1.1, 1.0, 1.0],
    )

    # ── Row 0: FPS bar + latency percentile bars ──────────────────────────────
    ax_fps  = fig.add_subplot(gs[0, :n//2])
    ax_perc = fig.add_subplot(gs[0, n//2:])

    mean_ms_vals = [r["mean_ms"] for r in results]
    bars = ax_fps.bar(labels, mean_ms_vals, color=colors, width=0.5, zorder=3)
    ax_fps.axhline(1000/30, color="#ff4444", lw=1.2, ls="--", label="33.3 ms (real-time)")
    ax_fps.set_title("Mean Solve Latency")
    ax_fps.set_ylabel("ms / frame")
    ax_fps.legend(fontsize=8)
    ax_fps.grid(axis="y", zorder=0)
    ax_fps.set_ylim(0, max(mean_ms_vals) * 1.35)
    for bar, v in zip(bars, mean_ms_vals):
        ax_fps.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.1f} ms", ha="center", va="bottom", fontsize=9, color="#eee")

    # Grouped percentile bars
    x = np.arange(n)
    w = 0.22
    for i, (key, alpha) in enumerate([("p50", 1.0), ("p95", 0.72), ("p99", 0.50)]):
        vals = [r[key] for r in results]
        bars2 = ax_perc.bar(x + (i-1)*w, vals, width=w,
                            color=colors, alpha=alpha, zorder=3,
                            label=f"p{key[1:]}")
    ax_perc.set_title("Latency Percentiles")
    ax_perc.set_ylabel("ms / frame")
    ax_perc.set_xticks(x)
    ax_perc.set_xticklabels(labels)
    ax_perc.legend(fontsize=8)
    ax_perc.grid(axis="y", zorder=0)

    # ── Row 1: Per-frame latency time series ──────────────────────────────────
    for col, (r, c) in enumerate(zip(results, colors)):
        ax = fig.add_subplot(gs[1, col])
        t_ms = r["latency_ms"]
        frames = np.arange(len(t_ms))

        ax.fill_between(frames, t_ms, alpha=0.25, color=c)
        ax.plot(frames, t_ms, lw=0.8, color=c)
        ax.axhline(r["p50"], color="white",   lw=0.9, ls="--", alpha=0.6, label=f"p50 {r['p50']:.1f}ms")
        ax.axhline(r["p95"], color="#ffcc44", lw=0.9, ls="--", alpha=0.7, label=f"p95 {r['p95']:.1f}ms")
        ax.set_title(r["label"])
        ax.set_xlabel("frame")
        if col == 0:
            ax.set_ylabel("latency (ms)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(zorder=0)
        ax.set_xlim(0, len(t_ms) - 1)

    # ── Row 2: Latency distribution (histogram) ───────────────────────────────
    for col, (r, c) in enumerate(zip(results, colors)):
        ax = fig.add_subplot(gs[2, col])
        t_ms = r["latency_ms"]
        ax.hist(t_ms, bins=30, color=c, alpha=0.85, edgecolor="#0f0f0f", lw=0.4, zorder=3)
        ax.axvline(r["p50"], color="white",   lw=1.0, ls="--", alpha=0.7, label=f"p50")
        ax.axvline(r["p95"], color="#ffcc44", lw=1.0, ls="--", alpha=0.8, label=f"p95")
        ax.axvline(r["p99"], color="#ff6644", lw=1.0, ls="--", alpha=0.8, label=f"p99")

        viol_pct = r["violations"].mean() * 100
        info = (f"mean {r['mean_ms']:.1f} ms  |  {r['fps']:.0f} fps\n"
                f"converged {r['converge_rate']*100:.0f}%  |  "
                f"joint viol {viol_pct:.1f}%")
        ax.set_title(f"{r['label']} — distribution")
        ax.set_xlabel("latency (ms)")
        if col == 0:
            ax.set_ylabel("frame count")
        ax.legend(fontsize=7)
        ax.grid(axis="y", zorder=0)
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#bbb",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111", alpha=0.7))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot_metrics] saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GMR timing metrics for all clips.")
    parser.add_argument("--clips", nargs="+", default=CLIPS,
                        help="Clips to evaluate (default: all four)")
    parser.add_argument("--output", default="outputs/metrics_timing.png")
    args = parser.parse_args()

    results = []
    for clip in args.clips:
        print(f"[plot_metrics] running {clip} …")
        t0 = time.perf_counter()
        r = run_clip(clip)
        elapsed = time.perf_counter() - t0
        print(f"  {r['num_frames']} frames  |  GMR: {r['fps']:.0f} fps  "
              f"p50={r['p50']:.1f}ms  p95={r['p95']:.1f}ms  "
              f"(wall {r['gmr_wall_s']:.2f}s)")
        results.append(r)

    print("[plot_metrics] generating figure …")
    plot(results, pathlib.Path(args.output))


if __name__ == "__main__":
    main()
