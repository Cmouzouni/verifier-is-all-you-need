"""Generate ARC reversal figures for the paper.

Outputs:
  fig10_arc_ranking.pdf      — six-bar chart of all ARC architectures
  fig11_synth_reversal.pdf   — side-by-side: synth helps on Phase A, hurts on ARC
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
# FIGURE 10: ARC architecture ranking
# ════════════════════════════════════════════════════════════════════════

def fig_arc_ranking():
    state = json.loads(Path("results/arc_22b_attack/state.json").read_text())

    # Hand-curated ordering and labels
    # qwen-22b single baseline re-measured: 34/150 = 22.7% (Wilson 95% CI [16.7%, 30.0%])
    bars = [
        ("qwen-22b\nsingle (n=150)", 34, 150, "baseline"),
        ("4× qwen-22b\n+ LLM synth", 4, 30, "synth"),
        ("8× qwen-22b\n+ LLM synth", 6, 30, "synth"),
        ("4×22b + 4×llama\n+ LLM synth", 3, 30, "synth"),
        ("8× qwen-22b\nmajority vote", 10, 30, "consensus"),
        ("8× qwen-22b\nbest-of-8 (oracle)", 12, 30, "oracle"),
        ("16× qwen-22b\nbest-of-16 (oracle)", 13, 30, "oracle"),
    ]
    # Re-order: synth (low) → baseline → consensus → oracles
    # Sort by rate
    bars.sort(key=lambda x: x[1] / x[2])

    labels = [b[0] for b in bars]
    rates = [b[1] / b[2] for b in bars]
    types = [b[3] for b in bars]
    cis = [wilson_ci(b[1], b[2]) for b in bars]
    err_low = [r - lo for r, (lo, _) in zip(rates, cis)]
    err_high = [hi - r for r, (_, hi) in zip(rates, cis)]

    color_map = {
        "synth": "#e41a1c",       # red — synth hurts on ARC
        "baseline": "#969696",    # gray — single-model baseline
        "consensus": "#4daf4a",   # green — majority vote works
        "oracle": "#377eb8",      # blue — oracle ceiling
    }
    colors = [color_map[t] for t in types]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(labels))
    bars_obj = ax.bar(x, rates, yerr=[err_low, err_high], capsize=3,
                       color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(34/150, color="#969696", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(len(bars) - 0.5, 34/150 + 0.005, "single-model baseline (22.7%, n=150)",
            ha="right", va="bottom", fontsize=7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("ARC-30 correctness (Wilson 95% CI)")
    ax.set_ylim(0, 0.60)
    ax.set_title("ARC architecture ranking — qwen-22b throughout", pad=12)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values
    for i, (b, r) in enumerate(zip(bars_obj, rates)):
        ax.text(i, r + (err_high[i] if err_high[i] else 0.02) + 0.005,
                f"{r:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["oracle"], edgecolor="black", label="Oracle best-of-N (ceiling)"),
        Patch(facecolor=color_map["consensus"], edgecolor="black", label="Majority vote (consensus)"),
        Patch(facecolor=color_map["baseline"], edgecolor="black", label="Single-model baseline"),
        Patch(facecolor=color_map["synth"], edgecolor="black", label="LLM-synthesized game"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_arc_ranking.pdf")
    plt.close(fig)
    print("✓ fig10_arc_ranking.pdf")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 11: synth reversal (Phase A vs GSM8K vs ARC)
# ════════════════════════════════════════════════════════════════════════

def fig_synth_reversal():
    """Show that the synth effect REVERSES from Phase A → ARC."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Each domain has (majority_vote_rate, synth_rate)
    domains = [
        ("Phase A\n(simple math)", 0.708, 0.896),  # n=250 from synth ablation
        ("GSM8K\n(word problems)", 0.580, 0.870),  # qwen-7b: 58%→87% via synth-completed game
        ("ARC\n(visual grids)", 0.333, 0.133),     # majvote 33%, 4×22b synth 13.3%
    ]

    n_domains = len(domains)
    width = 0.35
    x = np.arange(n_domains)

    mv_rates = [d[1] for d in domains]
    synth_rates = [d[2] for d in domains]

    bars1 = ax.bar(x - width / 2, mv_rates, width, label="Majority vote",
                    color="#4daf4a", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, synth_rates, width, label="LLM synthesizer",
                    color="#377eb8", edgecolor="black", linewidth=0.5)

    # Annotate deltas
    for i, ((label, mv, syn), b1, b2) in enumerate(zip(domains, bars1, bars2)):
        delta = (syn - mv) * 100
        sign = "+" if delta >= 0 else ""
        color = "#2ca02c" if delta > 0 else "#d62728"
        ax.text(i, max(mv, syn) + 0.04, f"Δ = {sign}{delta:.1f}pp",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
        # bar values
        ax.text(b1.get_x() + b1.get_width() / 2, mv + 0.005, f"{mv:.0%}",
                ha="center", va="bottom", fontsize=7)
        ax.text(b2.get_x() + b2.get_width() / 2, syn + 0.005, f"{syn:.0%}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in domains], fontsize=9)
    ax.set_ylabel("Correctness")
    ax.set_ylim(0, 1.05)
    ax.set_title("The verifiability boundary: synth effect reverses on hard-to-verify outputs",
                 pad=8, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    # Vertical separator marking the boundary
    ax.axvline(1.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(1.5, 1.0, "verifiability\nboundary", ha="center", va="top",
            fontsize=8, color="#555555", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_synth_reversal.pdf")
    plt.close(fig)
    print("✓ fig11_synth_reversal.pdf")


def main():
    fig_arc_ranking()
    fig_synth_reversal()


if __name__ == "__main__":
    main()
