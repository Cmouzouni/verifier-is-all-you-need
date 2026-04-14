"""Generate the cost-correctness Pareto plot for the AlphaProgram paper.

Reads validation block result JSONs and produces:
  pivot-a/alpha_program/figures/pareto_arc_agi_1.pdf
  pivot-a/alpha_program/figures/pareto_arc_agi_1.png

The plot is the headline figure: x-axis = $/task (log), y-axis =
correctness on ARC-AGI-1 public eval, each point a model/architecture.
AlphaProgram should land above-and-left of every published reasoner at
fraction of cost.
"""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from alpha_program.analyze_validation import (
    parse_validation_run,
    parse_frontier_run,
    PUBLISHED_REFERENCE,
    wilson_ci,
)


# Plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    fig, ax = plt.subplots(figsize=(7, 5))

    output_dir = RESULTS_DIR / "alpha_program"

    # ── AlphaProgram runs (our points) ────────────────────────────────────
    alpha_runs = []  # list of (label, x_cost, y_rate, ci_lo, ci_hi, marker_kwargs)

    # E2-bis (training set; show as a "training pilot" point with different style)
    e2 = parse_validation_run(output_dir / "e2_verifier_loop_qwen22b.json", k_per_task=32)
    if e2:
        alpha_runs.append(dict(
            label=f"AlphaProgram (E2-bis training-30, K=32)\n{e2.k_correct}/{e2.n_tasks}",
            x=e2.cost_per_task, y=e2.rate, lo=e2.ci_lo, hi=e2.ci_hi,
            marker="s", color="#666666", facecolor="white", edgecolor="#666666",
        ))

    # V1 (eval set, the canonical headline)
    v1_files = [
        ("v1_arc1_eval_qwen22b_n100_k16.json", 16),
        ("v1_arc1_eval_qwen22b_n100.json", 32),
        ("v1_arc1_eval_qwen22b.json", 32),
    ]
    for fname, k in v1_files:
        v1 = parse_validation_run(output_dir / fname, k_per_task=k)
        if v1:
            alpha_runs.append(dict(
                label=f"AlphaProgram V1 (eval-{v1.n_tasks}, K={k})\n{v1.k_correct}/{v1.n_tasks}",
                x=v1.cost_per_task, y=v1.rate, lo=v1.ci_lo, hi=v1.ci_hi,
                marker="*", color="#d62728", facecolor="#d62728", edgecolor="black",
                markersize=200,
            ))
            break

    # ── Same-task frontier baselines (V2a, V2b) ───────────────────────────
    same_task = []
    for fname in ["v2a_deepseek_v31_eval400_4k.json", "v2a_deepseek_v31_eval400.json"]:
        v2a = parse_frontier_run(output_dir / fname)
        if v2a:
            same_task.append(dict(
                label=f"DeepSeek-V3.1 (same eval set)\n{v2a.k_correct}/{v2a.n_tasks}",
                x=v2a.cost_per_task, y=v2a.rate, lo=v2a.ci_lo, hi=v2a.ci_hi,
                marker="o", color="#1f77b4",
            ))
            break

    # ── Published reference points ────────────────────────────────────────
    ref_points = []
    for label, arch, n, rate, ct, source in PUBLISHED_REFERENCE:
        if ct is None:
            continue  # skip unknown-cost points
        lo, hi = wilson_ci(int(rate * n), n)
        ref_points.append(dict(label=label, x=ct, y=rate, lo=lo, hi=hi, source=source))

    # ── Plot ──────────────────────────────────────────────────────────────
    # AlphaProgram as red star
    for r in alpha_runs:
        err_lo = r["y"] - r["lo"]
        err_hi = r["hi"] - r["y"]
        ax.errorbar(
            r["x"], r["y"],
            yerr=[[err_lo], [err_hi]],
            fmt="none",
            ecolor=r.get("edgecolor", r.get("color", "#d62728")),
            elinewidth=1.0,
            capsize=3,
            alpha=0.8,
        )
        ax.scatter(
            r["x"], r["y"],
            s=r.get("markersize", 130),
            marker=r["marker"],
            facecolor=r.get("facecolor", r.get("color", "#d62728")),
            edgecolor=r.get("edgecolor", "black"),
            linewidth=1.2,
            zorder=10,
            label=r["label"],
        )

    # Same-task baselines
    for r in same_task:
        err_lo = r["y"] - r["lo"]
        err_hi = r["hi"] - r["y"]
        ax.errorbar(
            r["x"], r["y"],
            yerr=[[err_lo], [err_hi]],
            fmt="none",
            ecolor=r["color"],
            elinewidth=1.0,
            capsize=3,
            alpha=0.6,
        )
        ax.scatter(
            r["x"], r["y"],
            s=80,
            marker=r["marker"],
            facecolor=r["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
            label=r["label"],
        )

    # Published reference points
    for r in ref_points:
        ax.scatter(
            r["x"], r["y"],
            s=60, marker="^", color="#888888",
            edgecolor="black", linewidth=0.6,
            alpha=0.8, zorder=5,
        )
        ax.annotate(
            r["label"],
            xy=(r["x"], r["y"]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7, color="#444444",
            verticalalignment="center",
        )

    # Pareto frontier (visual aid)
    all_points = [(r["x"], r["y"]) for r in alpha_runs + same_task + ref_points]
    if all_points:
        all_points.sort(key=lambda p: p[0])
        pareto_x, pareto_y = [], []
        max_y = -1
        for x, y in all_points:
            if y > max_y:
                pareto_x.append(x)
                pareto_y.append(y)
                max_y = y
        if len(pareto_x) >= 2:
            ax.plot(pareto_x, pareto_y, "k--", alpha=0.3, linewidth=1.0, label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (USD, log scale)")
    ax.set_ylabel("ARC-AGI-1 correctness (Wilson 95% CI)")
    ax.set_ylim(0, 0.7)
    ax.set_title("Cost vs correctness on ARC-AGI-1: AlphaProgram (cheap collective + symbolic verifier)\nvs frontier reasoners on the same / canonical public eval set", pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto_arc_agi_1.pdf")
    fig.savefig(FIG_DIR / "pareto_arc_agi_1.png")
    plt.close(fig)
    print(f"✓ {FIG_DIR / 'pareto_arc_agi_1.pdf'}")
    print(f"✓ {FIG_DIR / 'pareto_arc_agi_1.png'}")


if __name__ == "__main__":
    main()
