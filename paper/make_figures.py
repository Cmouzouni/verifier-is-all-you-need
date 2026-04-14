"""Generate publication-quality figures from the full sweep results.

Outputs:
  fig1_pareto.pdf       — cost vs correctness Pareto frontier with baselines
  fig2_topology.pdf     — topology comparison bar chart
  fig3_gamma_response.pdf — diversity D(γ) and correctness vs γ
  fig4_model_x_topology.pdf — heatmap: model config × topology correctness
  fig5_cost_dynamics.pdf — per-model cost breakdown
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────
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

COLORS = {
    "all-7b": "#1f77b4",
    "all-22b": "#ff7f0e",
    "propose22b-crit7b": "#2ca02c",
    "propose7b-synth22b": "#d62728",
    "propose17b-crit7b": "#9467bd",
    "qwen-7b": "#7fb3d5",
    "qwen-22b": "#e59866",
    "qwen-17b": "#bb8fce",
}

TOPOLOGY_ORDER = [
    "propose-only",
    "propose-critique",
    "consensus-insight",
    "propose-critique-synth",
    "full-debate",
    "all-insight-synth",
]

TOPOLOGY_LABELS = {
    "propose-only": "Propose only",
    "propose-critique": "Propose+Critique",
    "consensus-insight": "Consensus+Insight",
    "propose-critique-synth": "Critique+Synth",
    "full-debate": "Full Debate",
    "all-insight-synth": "Insight+Synth",
}

MODEL_LABELS = {
    "all-7b": "all-7B",
    "all-22b": "all-22B",
    "propose22b-crit7b": "22B prop / 7B crit",
    "propose7b-synth22b": "7B prop / 22B synth",
    "propose17b-crit7b": "397B prop / 7B crit",
}

GAMMA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def load_data():
    data = json.load(open("pivot-a/results/full_sweep/results.json"))
    cache = json.load(open("pivot-a/results/cached_results.json"))
    return data, cache


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - half), min(1, center + half)


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: Cost vs Correctness Pareto Frontier
# ════════════════════════════════════════════════════════════════════════
def fig_pareto(data, cache):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Collect (cost, correctness) per condition
    by_cond = defaultdict(list)
    for e in data["episodes"]:
        by_cond[(e["model_config"], e["topology"], e["gamma"])].append(e)

    # Plot each model config as scatter
    for mc in MODEL_LABELS:
        xs, ys = [], []
        for (mc_, tp, g), eps in by_cond.items():
            if mc_ != mc:
                continue
            cost = sum(e["total_cost_usd"] for e in eps) / len(eps)
            rate = sum(e["correct"] for e in eps) / len(eps)
            xs.append(cost)
            ys.append(rate)
        ax.scatter(xs, ys, label=MODEL_LABELS[mc], alpha=0.55, s=22, color=COLORS[mc], edgecolors="none")

    # Plot baselines
    for model, b in cache["baselines"].items():
        ax.scatter(b["cost"] / b["n_total"], b["correct_rate"],
                   marker="*", s=240, color=COLORS[model], edgecolors="black", linewidths=1.0,
                   label=f"{model} (single)", zorder=5)

    # Pareto frontier (best correctness at each cost level)
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
    ax.set_title("Cost–Performance Pareto Frontier", pad=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9, fontsize=7)

    fig.tight_layout()
    fig.savefig("pivot-a/paper/figures/fig1_pareto.pdf")
    plt.close(fig)
    print("✓ fig1_pareto.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Topology comparison
# ════════════════════════════════════════════════════════════════════════
def fig_topology(data, cache):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    by_top = defaultdict(list)
    for e in data["episodes"]:
        by_top[e["topology"]].append(e)

    rates, ci_lows, ci_highs, costs = [], [], [], []
    for tp in TOPOLOGY_ORDER:
        eps = by_top[tp]
        k = sum(e["correct"] for e in eps)
        n = len(eps)
        rates.append(k / n)
        lo, hi = wilson_ci(k, n)
        ci_lows.append(lo)
        ci_highs.append(hi)
        costs.append(sum(e["total_cost_usd"] for e in eps) / n)

    x = np.arange(len(TOPOLOGY_ORDER))
    err_low = [r - lo for r, lo in zip(rates, ci_lows)]
    err_high = [hi - r for r, hi in zip(rates, ci_highs)]

    bars = ax.bar(x, rates, yerr=[err_low, err_high], capsize=3,
                   color=["#bdd7e7", "#bdd7e7", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
                   edgecolor="black", linewidth=0.5)

    # Baseline reference lines
    ax.axhline(cache["baselines"]["qwen-7b"]["correct_rate"], color="gray", linestyle=":",
               linewidth=1.0, alpha=0.7, label=f"qwen-7B single ({cache['baselines']['qwen-7b']['correct_rate']:.0%})")
    ax.axhline(cache["baselines"]["qwen-22b"]["correct_rate"], color="orange", linestyle=":",
               linewidth=1.0, alpha=0.7, label=f"qwen-22B single ({cache['baselines']['qwen-22b']['correct_rate']:.0%})")
    ax.axhline(cache["baselines"]["qwen-17b"]["correct_rate"], color="red", linestyle=":",
               linewidth=1.0, alpha=0.7, label=f"qwen-397B single ({cache['baselines']['qwen-17b']['correct_rate']:.0%})")

    ax.set_xticks(x)
    ax.set_xticklabels([TOPOLOGY_LABELS[t] for t in TOPOLOGY_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Correctness (Wilson 95% CI)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Topology effect on correctness (averaged over γ and model configs)", pad=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)

    fig.tight_layout()
    fig.savefig("pivot-a/paper/figures/fig2_topology.pdf")
    plt.close(fig)
    print("✓ fig2_topology.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: γ Response (diversity + correctness)
# ════════════════════════════════════════════════════════════════════════
def fig_gamma_response(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    by_top_gamma = defaultdict(list)
    for e in data["episodes"]:
        by_top_gamma[(e["topology"], e["gamma"])].append(e)

    # Left: Diversity D(γ)
    for tp in TOPOLOGY_ORDER[-3:]:  # focus on top 3 topologies
        ds = []
        for g in GAMMA_VALUES:
            eps = by_top_gamma[(tp, g)]
            ds.append(sum(e["diversity_score"] for e in eps) / len(eps) if eps else 0)
        axes[0].plot(GAMMA_VALUES, ds, marker="o", label=TOPOLOGY_LABELS[tp], linewidth=1.5, markersize=5)

    # MFG prediction (uniform max for N=4, K=4 frameworks)
    axes[0].axhline(0.75, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="theoretical max (1−1/K)")

    axes[0].set_xlabel("Congestion parameter γ")
    axes[0].set_ylabel("Diversity D = 1 − max(µᵢ)")
    axes[0].set_xscale("symlog", linthresh=0.5)
    axes[0].set_ylim(0.3, 0.85)
    axes[0].set_title("(a) Diversity response", pad=6)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9)

    # Right: Correctness C(γ) by topology
    for tp in TOPOLOGY_ORDER:
        rates = []
        ci_lows = []
        ci_highs = []
        for g in GAMMA_VALUES:
            eps = by_top_gamma[(tp, g)]
            k = sum(e["correct"] for e in eps)
            n = len(eps)
            r = k / n if n else 0
            rates.append(r)
            lo, hi = wilson_ci(k, n)
            ci_lows.append(lo)
            ci_highs.append(hi)
        axes[1].plot(GAMMA_VALUES, rates, marker="o", label=TOPOLOGY_LABELS[tp], linewidth=1.3, markersize=4)

    axes[1].set_xlabel("Congestion parameter γ")
    axes[1].set_ylabel("Correctness")
    axes[1].set_xscale("symlog", linthresh=0.5)
    axes[1].set_ylim(0.55, 1.0)
    axes[1].set_title("(b) Correctness by topology", pad=6)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=6, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig("pivot-a/paper/figures/fig3_gamma_response.pdf")
    plt.close(fig)
    print("✓ fig3_gamma_response.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: Model × Topology heatmap
# ════════════════════════════════════════════════════════════════════════
def fig_heatmap(data):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    by_mc_tp = defaultdict(list)
    for e in data["episodes"]:
        by_mc_tp[(e["model_config"], e["topology"])].append(e)

    mcs = list(MODEL_LABELS.keys())
    matrix = np.zeros((len(mcs), len(TOPOLOGY_ORDER)))
    for i, mc in enumerate(mcs):
        for j, tp in enumerate(TOPOLOGY_ORDER):
            eps = by_mc_tp[(mc, tp)]
            if eps:
                matrix[i, j] = sum(e["correct"] for e in eps) / len(eps)

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(TOPOLOGY_ORDER)))
    ax.set_xticklabels([TOPOLOGY_LABELS[t] for t in TOPOLOGY_ORDER], rotation=18, ha="right")
    ax.set_yticks(range(len(mcs)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in mcs])

    # Annotate cells
    for i in range(len(mcs)):
        for j in range(len(TOPOLOGY_ORDER)):
            color = "white" if matrix[i, j] < 0.7 else "black"
            ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Correctness", fontsize=8)

    ax.set_title("Model config × Topology (averaged across γ)", pad=8)
    fig.tight_layout()
    fig.savefig("pivot-a/paper/figures/fig4_heatmap.pdf")
    plt.close(fig)
    print("✓ fig4_heatmap.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 5: Best-of-each — winning configurations
# ════════════════════════════════════════════════════════════════════════
def fig_winners(data, cache):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    # For each unique config, find its best (cost, correctness) point
    by_cond = defaultdict(list)
    for e in data["episodes"]:
        by_cond[(e["model_config"], e["topology"], e["gamma"])].append(e)

    # Group by (model_config, topology) - take best gamma
    by_mc_tp = defaultdict(list)
    for (mc, tp, g), eps in by_cond.items():
        rate = sum(e["correct"] for e in eps) / len(eps)
        cost = sum(e["total_cost_usd"] for e in eps) / len(eps)
        by_mc_tp[(mc, tp)].append((g, rate, cost))

    # Best 100% configs sorted by cost
    perfect = []
    for (mc, tp, g), eps in by_cond.items():
        rate = sum(e["correct"] for e in eps) / len(eps)
        if rate == 1.0 and len(eps) >= 12:
            cost = sum(e["total_cost_usd"] for e in eps) / len(eps)
            perfect.append((mc, tp, g, cost))
    perfect.sort(key=lambda x: x[3])

    # Show top 10 cheapest 100% configurations
    perfect = perfect[:10]
    labels = [f"{MODEL_LABELS[mc]}\n{TOPOLOGY_LABELS[tp]} (γ={g})" for mc, tp, g, c in perfect]
    costs = [c for _, _, _, c in perfect]
    colors = [COLORS[mc] for mc, _, _, _ in perfect]

    y = np.arange(len(perfect))
    bars = ax.barh(y, costs, color=colors, edgecolor="black", linewidth=0.4)

    # Reference: baselines
    qwen_22b_cost = cache["baselines"]["qwen-22b"]["cost"] / cache["baselines"]["qwen-22b"]["n_total"]
    qwen_17b_cost = cache["baselines"]["qwen-17b"]["cost"] / cache["baselines"]["qwen-17b"]["n_total"]
    ax.axvline(qwen_22b_cost, color="orange", linestyle="--", linewidth=1.0,
               label=f"qwen-22B single (87% correct, ${qwen_22b_cost:.4f}/problem)")
    ax.axvline(qwen_17b_cost, color="red", linestyle="--", linewidth=1.0,
               label=f"qwen-397B single (70% correct, ${qwen_17b_cost:.4f}/problem)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Cost per problem (USD)")
    ax.set_xscale("log")
    ax.set_title("Cheapest configurations achieving 100% correctness (n=15 episodes each)", pad=8)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate cost values
    for i, (b, c) in enumerate(zip(bars, costs)):
        ax.text(c * 1.05, i, f"${c:.5f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig("pivot-a/paper/figures/fig5_winners.pdf")
    plt.close(fig)
    print("✓ fig5_winners.pdf")


# ════════════════════════════════════════════════════════════════════════
# COMPUTE ALL STATISTICS FOR THE PAPER
# ════════════════════════════════════════════════════════════════════════
def compute_paper_stats(data, cache):
    """Compute and print all statistics referenced in the paper."""
    stats = {}

    # 1. Baselines
    print("\n=== BASELINES ===")
    for m, b in cache["baselines"].items():
        rate = b["correct_rate"]
        n = b["n_total"]
        cost = b["cost"]
        cost_per = cost / n
        lo, hi = wilson_ci(int(rate * n), n)
        print(f"  {m:12s}: {rate:.1%} [{lo:.1%}, {hi:.1%}] cost=${cost:.4f} (${cost_per:.5f}/prob)")
        stats[f"baseline_{m}"] = {"rate": rate, "n": n, "ci": [lo, hi], "cost_total": cost, "cost_per": cost_per}

    # 2. Aggregate γ response
    print("\n=== AGGREGATE γ RESPONSE ===")
    by_g = defaultdict(list)
    for e in data["episodes"]:
        by_g[e["gamma"]].append(e)
    for g in sorted(by_g.keys()):
        eps = by_g[g]
        k = sum(e["correct"] for e in eps)
        n = len(eps)
        d = sum(e["diversity_score"] for e in eps) / n
        lo, hi = wilson_ci(k, n)
        print(f"  γ={g:4.1f}: D={d:.3f} correct={k/n:.1%} [{lo:.1%}, {hi:.1%}] (n={n})")

    # 3. Best 100% configurations
    print("\n=== TOP 100% CONFIGS BY COST ===")
    by_cond = defaultdict(list)
    for e in data["episodes"]:
        by_cond[(e["model_config"], e["topology"], e["gamma"])].append(e)
    perfect = []
    for cond, eps in by_cond.items():
        rate = sum(e["correct"] for e in eps) / len(eps)
        if rate == 1.0 and len(eps) >= 12:
            cost = sum(e["total_cost_usd"] for e in eps) / len(eps)
            perfect.append((cond, cost, len(eps)))
    perfect.sort(key=lambda x: x[1])
    for cond, cost, n in perfect[:8]:
        print(f"  {cond[0]:22s} × {cond[1]:25s} γ={cond[2]:.1f}: ${cost:.5f} (n={n})")

    # 4. Cost ratio (game vs frontier)
    print("\n=== COST RATIO ===")
    if perfect:
        cheapest_perfect_cost = perfect[0][1]
        frontier_cost = cache["baselines"]["qwen-17b"]["cost"] / 30
        ratio = frontier_cost / cheapest_perfect_cost
        print(f"  Cheapest 100% config: ${cheapest_perfect_cost:.5f}/problem")
        print(f"  Frontier (qwen-17b/397B): ${frontier_cost:.5f}/problem")
        print(f"  Cost ratio: frontier is {ratio:.0f}x more expensive")
        print(f"  ... and the frontier achieves only 70.0% vs the game's 100%")

    # 5. Topology summary
    print("\n=== TOPOLOGY SUMMARY (averaged over γ, model configs) ===")
    by_tp = defaultdict(list)
    for e in data["episodes"]:
        by_tp[e["topology"]].append(e)
    for tp in TOPOLOGY_ORDER:
        eps = by_tp[tp]
        k = sum(e["correct"] for e in eps)
        n = len(eps)
        cost = sum(e["total_cost_usd"] for e in eps) / n
        lo, hi = wilson_ci(k, n)
        print(f"  {tp:25s}: {k/n:.1%} [{lo:.1%},{hi:.1%}] cost=${cost:.5f} (n={n})")

    return stats


def main():
    Path("pivot-a/paper/figures").mkdir(parents=True, exist_ok=True)
    data, cache = load_data()
    print(f"Loaded {len(data['episodes'])} episodes from {len(set((e['model_config'],e['topology'],e['gamma']) for e in data['episodes']))} conditions")

    fig_pareto(data, cache)
    fig_topology(data, cache)
    fig_gamma_response(data)
    fig_heatmap(data)
    fig_winners(data, cache)

    stats = compute_paper_stats(data, cache)


if __name__ == "__main__":
    main()
