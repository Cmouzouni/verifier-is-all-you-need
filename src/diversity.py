"""Diversity and statistical metrics for experiment analysis."""

from __future__ import annotations

import math
from collections import Counter


def diversity_score(assignments: list[str]) -> float:
    """D = 1 - max_i(µ_i). Measures population spread across frameworks."""
    if not assignments:
        return 0.0
    counts = Counter(assignments)
    n = len(assignments)
    return 1.0 - max(counts.values()) / n


def framework_coverage(assignments: list[str], total_frameworks: int) -> float:
    """Fraction of available frameworks that have at least one agent."""
    if not assignments or total_frameworks == 0:
        return 0.0
    return len(set(assignments)) / total_frameworks


def shannon_entropy(assignments: list[str]) -> float:
    """Shannon entropy of the assignment distribution."""
    if not assignments:
        return 0.0
    counts = Counter(assignments)
    n = len(assignments)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def max_entropy(n_frameworks: int) -> float:
    """Maximum entropy for uniform distribution over n frameworks."""
    if n_frameworks <= 1:
        return 0.0
    return math.log2(n_frameworks)


def normalized_entropy(assignments: list[str], n_frameworks: int) -> float:
    """Entropy normalized to [0, 1] by dividing by max entropy."""
    h_max = max_entropy(n_frameworks)
    if h_max == 0:
        return 0.0
    return shannon_entropy(assignments) / h_max


def mfg_predicted_diversity(
    gamma: float,
    n_agents: int = 4,
    n_frameworks: int = 4,
    tau: float = 0.3,
) -> float:
    """Approximate MFG-predicted diversity D(γ) for equal base values.

    At γ=0 with equal values: uniform selection → D ≈ 1 - 1/K for large N,
    but with small N the max occupancy is higher due to sampling variance.

    At γ→∞: near-perfect round-robin → D = 1 - 1/K.

    This is a Monte Carlo estimate used as the theoretical reference curve.
    """
    import random

    rng = random.Random(42)
    n_trials = 2000
    total_d = 0.0

    for _ in range(n_trials):
        from .congestion import assign_frameworks
        assignments, _ = assign_frameworks(
            frameworks=[f"f{i}" for i in range(n_frameworks)],
            n_agents=n_agents,
            gamma=gamma,
            tau=tau,
            seed=rng.randint(0, 2**31),
        )
        total_d += diversity_score(assignments)

    return total_d / n_trials
