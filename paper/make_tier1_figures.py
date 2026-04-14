"""Generate Tier 1 figures for the revised paper.

Outputs:
  fig6_gsm8k.pdf            — GSM8K generalization (cheap collective lifts 58→87%)
  fig7_synthesis_ablation.pdf — Three aggregator conditions
  fig8_capability_ceiling.pdf — Family replication: game saturates at single-model capability
  fig9_pareto_revised.pdf    — Updated Pareto with fair frontier baseline
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse style from make_figures.py
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - half), min(1, center + half)


# ════════════════════════════════════════════════════════════════════════
# FIGURE 6: GSM8K results
# ════════════════════════════════════════════════════════════════════════
def fig_gsm8k():
    data = json.load(open("results/tier1_gsm8k/results.json"))

    # Baselines
    bls = data["baselines"]
    bl_models = ["qwen-7b", "qwen-22b", "qwen-17b"]
    bl_rates = []
    bl_ns = []
    for m in bl_models:
        recs = bls[m]
        k = sum(r["correct"] for r in recs)
        n = len(recs)
        bl_rates.append(k / n)
        bl_ns.append((k, n))

    # Game configs (from "game_episodes")
    by_label = defaultdict(list)
    for r in data["game_episodes"]:
        by_label[r["config_label"]].append(r)

    # Order by correctness
    game_data = []
    for label, recs in by_label.items():
        k = sum(r["correct"] for r in recs)
        n = len(recs)
        cost = sum(r["total_cost_usd"] for r in recs) / n
        game_data.append((label, k / n, k, n, cost))
    game_data.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # Stack: baselines first (light), then games (dark)
    all_labels = []
    all_rates = []
    all_lows = []
    all_highs = []
    all_colors = []

    for m, (k, n) in zip(bl_models, bl_ns):
        all_labels.append(f"{m} single")
        rate = k / n
        all_rates.append(rate)
        lo, hi = wilson_ci(k, n)
        all_lows.append(rate - lo)
        all_highs.append(hi - rate)
        all_colors.append("#a6cee3")  # light blue for baselines

    for label, rate, k, n, cost in game_data:
        # Shorten the label
        parts = label.split("|")
        short = f"{parts[0]} ×\n{parts[1]} ({parts[2]})"
        all_labels.append(short)
        all_rates.append(rate)
        lo, hi = wilson_ci(k, n)
        all_lows.append(rate - lo)
        all_highs.append(hi - rate)
        all_colors.append("#1f78b4")  # dark blue for game configs

    x = np.arange(len(all_labels))
    bars = ax.bar(x, all_rates, yerr=[all_lows, all_highs], capsize=3,
                  color=all_colors, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("GSM8K-50 correctness (Wilson 95% CI)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("GSM8K generalization: cheap-7B collective lifts from 58% → 87%", pad=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate top performers
    for i, (rate, color) in enumerate(zip(all_rates, all_colors)):
        if color == "#1f78b4":  # game config
            ax.text(i, rate + (all_highs[i] if all_highs[i] else 0.02) + 0.005,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#a6cee3", edgecolor="black", label="Single-model baselines"),
        Patch(facecolor="#1f78b4", edgecolor="black", label="Game configurations"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_gsm8k.pdf")
    plt.close(fig)
    print("✓ fig6_gsm8k.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 7: Synthesis ablation
# ════════════════════════════════════════════════════════════════════════
def fig_synthesis_ablation():
    data = json.load(open("results/tier1_synthesis/results.json"))
    rates = data["rates"]
    costs = data["costs"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    labels = [
        "A. Majority vote\n(no synth)",
        "B. LLM synth\n(answers only)",
        "C. LLM synth\n(full reasoning)",
    ]
    keys = ["majority_vote", "synth_answers_only", "synth_full_reasoning"]
    rs = [rates[k] for k in keys]
    cs = [costs[k] for k in keys]

    n_episodes = data["n_episodes"]
    cis = [wilson_ci(int(r * n_episodes), n_episodes) for r in rs]
    err_low = [r - lo for r, (lo, _) in zip(rs, cis)]
    err_high = [hi - r for r, (_, hi) in zip(rs, cis)]

    x = np.arange(3)
    colors = ["#fbb4ae", "#b3cde3", "#1f78b4"]
    bars = ax.bar(x, rs, yerr=[err_low, err_high], capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)

    # Annotate gaps
    ax.annotate("", xy=(0.95, rs[1] - 0.005), xytext=(0.05, rs[0] + 0.005),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(0.5, (rs[0] + rs[1]) / 2 + 0.01,
            f"+{(rs[1] - rs[0]) * 100:.1f}pp\n(LLM\naggregation)",
            ha="center", va="center", fontsize=7, fontweight="bold")

    ax.annotate("", xy=(1.95, rs[2] - 0.005), xytext=(1.05, rs[1] + 0.005),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(1.5, (rs[1] + rs[2]) / 2 + 0.01,
            f"+{(rs[2] - rs[1]) * 100:.1f}pp\n(reading\nreasoning)",
            ha="center", va="center", fontsize=7, fontweight="bold")

    # Bar value labels
    for b, r, c in zip(bars, rs, cs):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.005,
                f"{r:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.55,
                f"${c:.5f}/ep", ha="center", va="bottom", fontsize=6, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Correctness (Wilson 95% CI)")
    ax.set_ylim(0.55, 1.0)
    ax.set_title(f"Synthesis ablation (n={n_episodes} episodes)", pad=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_synthesis_ablation.pdf")
    plt.close(fig)
    print("✓ fig7_synthesis_ablation.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 8: Capability ceiling (family replication)
# ════════════════════════════════════════════════════════════════════════
def fig_capability_ceiling():
    """Show that the game lifts cheap models to the ceiling but doesn't push past it."""

    # Phase A data
    family_data = json.load(open("results/tier1_family/results.json"))
    main_data = json.load(open("results/full_sweep/results.json"))
    cache = json.load(open("results/cached_results.json"))

    # Build the data: pairs of (single-model rate, game-with-that-model rate)
    # All on Phase A tasks
    rows = []

    # qwen-7b: from main sweep (all-7b × all-insight-synth × γ=1)
    qwen7b_baseline = cache["baselines"]["qwen-7b"]["correct_rate"]
    qwen7b_eps = [e for e in main_data["episodes"]
                  if e["model_config"] == "all-7b" and e["topology"] == "all-insight-synth" and e["gamma"] == 1.0]
    qwen7b_game = sum(e["correct"] for e in qwen7b_eps) / len(qwen7b_eps)
    rows.append(("Qwen-7B", qwen7b_baseline, qwen7b_game, len(qwen7b_eps)))

    # qwen-22b: similarly
    qwen22b_baseline = cache["baselines"]["qwen-22b"]["correct_rate"]
    qwen22b_eps = [e for e in main_data["episodes"]
                   if e["model_config"] == "all-22b" and e["topology"] == "all-insight-synth" and e["gamma"] == 1.0]
    qwen22b_game = sum(e["correct"] for e in qwen22b_eps) / len(qwen22b_eps)
    rows.append(("Qwen-22B (235B/22A)", qwen22b_baseline, qwen22b_game, len(qwen22b_eps)))

    # llama-70b: from family replication
    llama_baseline = family_data["results"]["baselines"]["llama-70b"]["rate"]
    llama_game = family_data["results"]["game"]["all-llama70b"]["rate"]
    llama_n = family_data["results"]["game"]["all-llama70b"]["n"]
    rows.append(("Llama-3.3-70B", llama_baseline, llama_game, llama_n))

    # deepseek: from family replication
    ds_baseline = family_data["results"]["baselines"]["deepseek"]["rate"]
    ds_game = family_data["results"]["game"]["all-deepseek"]["rate"]
    ds_n = family_data["results"]["game"]["all-deepseek"]["n"]
    rows.append(("DeepSeek-V3.1", ds_baseline, ds_game, ds_n))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    x = np.arange(len(rows))
    width = 0.35

    baselines = [r[1] for r in rows]
    games = [r[2] for r in rows]

    # Compute CIs
    bl_ns = [30, 30, 30, 30]  # baselines all on 30 Phase A tasks
    bl_cis = [wilson_ci(int(r * n), n) for r, n in zip(baselines, bl_ns)]
    bl_err_low = [r - lo for r, (lo, _) in zip(baselines, bl_cis)]
    bl_err_high = [hi - r for r, (_, hi) in zip(baselines, bl_cis)]

    g_cis = [wilson_ci(int(r * n), n) for r, n in zip(games, [r[3] for r in rows])]
    g_err_low = [r - lo for r, (lo, _) in zip(games, g_cis)]
    g_err_high = [hi - r for r, (_, hi) in zip(games, g_cis)]

    bars1 = ax.bar(x - width/2, baselines, width, yerr=[bl_err_low, bl_err_high],
                    label="Single-model baseline", color="#a6cee3", edgecolor="black", capsize=3)
    bars2 = ax.bar(x + width/2, games, width, yerr=[g_err_low, g_err_high],
                    label="Game (Insight+Synth, γ=1)", color="#1f78b4", edgecolor="black", capsize=3)

    # Annotate Δ
    for i, (label, b, g, n) in enumerate(rows):
        delta = g - b
        color = "green" if delta > 0 else "red"
        sign = "+" if delta > 0 else ""
        ax.text(i, max(b, g) + 0.04, f"{sign}{delta * 100:.1f}pp",
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=0, fontsize=8)
    ax.set_ylabel("Phase A correctness (Wilson 95% CI)")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("The capability ceiling: game lifts weak models, saturates at strong models",
                  pad=8, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_capability_ceiling.pdf")
    plt.close(fig)
    print("✓ fig8_capability_ceiling.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 9: Updated Pareto with fair frontier baseline
# ════════════════════════════════════════════════════════════════════════
def fig_pareto_revised():
    """Pareto frontier with the FAIR qwen-17b baseline (Tier 1.4)."""
    main_data = json.load(open("results/full_sweep/results.json"))
    cache = json.load(open("results/cached_results.json"))
    frontier_data = json.load(open("results/tier1_frontier/results.json"))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    COLORS = {
        "all-7b": "#1f77b4",
        "all-22b": "#ff7f0e",
        "propose22b-crit7b": "#2ca02c",
        "propose7b-synth22b": "#d62728",
        "propose17b-crit7b": "#9467bd",
    }
    LABELS = {
        "all-7b": "all-7B",
        "all-22b": "all-22B",
        "propose22b-crit7b": "22B prop / 7B crit",
        "propose7b-synth22b": "7B prop / 22B synth",
        "propose17b-crit7b": "397B prop / 7B crit",
    }

    by_cond = defaultdict(list)
    for e in main_data["episodes"]:
        by_cond[(e["model_config"], e["topology"], e["gamma"])].append(e)

    for mc in LABELS:
        xs, ys = [], []
        for (mc_, tp, g), eps in by_cond.items():
            if mc_ != mc:
                continue
            cost = sum(e["total_cost_usd"] for e in eps) / len(eps)
            rate = sum(e["correct"] for e in eps) / len(eps)
            xs.append(cost)
            ys.append(rate)
        ax.scatter(xs, ys, label=LABELS[mc], alpha=0.55, s=22, color=COLORS[mc], edgecolors="none")

    # Single baselines: original (with /no_think)
    for model, b in cache["baselines"].items():
        cost = b["cost"] / b["n_total"]
        ax.scatter(cost, b["correct_rate"], marker="*", s=240,
                   color="lightgray", edgecolors="black", linewidths=1.0,
                   label=f"{model} (no_think)" if model == "qwen-17b" else None, zorder=5)

    # FAIR frontier baseline (Tier 1.4)
    fair_phase_a_rate = frontier_data["phase_a"]["rate"]
    fair_phase_a_cost = frontier_data["phase_a"]["cost_per"]
    ax.scatter(fair_phase_a_cost, fair_phase_a_rate,
               marker="*", s=320, color="#bb8fce",
               edgecolors="black", linewidths=1.5,
               label=f"qwen-17b (with thinking, FAIR)", zorder=6)

    # Pareto frontier
    all_pts = []
    for (mc, tp, g), eps in by_cond.items():
        cost = sum(e["total_cost_usd"] for e in eps) / len(eps)
        rate = sum(e["correct"] for e in eps) / len(eps)
        all_pts.append((cost, rate))
    all_pts.sort()
    pareto = []
    best = -1
    for c, r in all_pts:
        if r > best:
            pareto.append((c, r))
            best = r
    if pareto:
        px, py = zip(*pareto)
        ax.step(px, py, where="post", color="black", linestyle="--", linewidth=1.2,
                alpha=0.6, label="Pareto frontier", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per problem (USD, log scale)")
    ax.set_ylabel("Correctness")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Pareto frontier: fair frontier comparison (Phase A)", pad=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_pareto_revised.pdf")
    plt.close(fig)
    print("✓ fig9_pareto_revised.pdf")


def main():
    fig_gsm8k()
    fig_synthesis_ablation()
    fig_capability_ceiling()
    fig_pareto_revised()


if __name__ == "__main__":
    main()
