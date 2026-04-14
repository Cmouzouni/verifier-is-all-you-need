"""Population-coupled congestion mechanism for framework selection.

The core MFG mechanism: sequential framework assignment with softmax routing
controlled by congestion parameter γ. At γ=0, all agents pick the highest-value
framework (herding). At γ>0, congestion penalizes crowded frameworks, producing
diversity.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class FrameworkState:
    """Tracks population distribution over frameworks during an episode."""
    frameworks: list[str]
    base_values: dict[str, float] = field(default_factory=dict)
    assignments: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.base_values:
            # Equal base values by default
            self.base_values = {f: 1.0 for f in self.frameworks}

    @property
    def occupancy(self) -> dict[str, float]:
        """Current occupancy µ_f for each framework."""
        n = len(self.assignments)
        if n == 0:
            return {f: 0.0 for f in self.frameworks}
        counts = {f: 0 for f in self.frameworks}
        for a in self.assignments:
            counts[a] += 1
        return {f: counts[f] / n for f in self.frameworks}

    @property
    def raw_counts(self) -> dict[str, int]:
        counts = {f: 0 for f in self.frameworks}
        for a in self.assignments:
            counts[a] += 1
        return counts

    def diversity_score(self) -> float:
        """D = 1 - max_i(µ_i). 0 = all agents on one framework, 0.75 = uniform over 4."""
        occ = self.occupancy
        if not occ:
            return 0.0
        return 1.0 - max(occ.values())

    def semantic_diversity(self) -> float:
        """Fraction of distinct frameworks used."""
        if not self.assignments:
            return 0.0
        return len(set(self.assignments)) / len(self.frameworks)


def select_framework(
    state: FrameworkState,
    gamma: float,
    tau: float = 0.3,
    epsilon: float = 1e-6,
    rng: random.Random | None = None,
) -> str:
    """Select next framework using population-coupled softmax.

    P(framework f) ∝ exp((V_f - γ × log(µ_f + ε)) / τ)

    At γ=0: pure value-based selection (all pick highest V_f, or uniform if equal).
    At γ>0: congestion penalty pushes agents away from crowded frameworks.

    Args:
        state: Current population state.
        gamma: Congestion parameter (≥0).
        tau: Softmax temperature.
        epsilon: Smoothing for log(µ_f).
        rng: Random generator for reproducibility.

    Returns:
        Selected framework name.
    """
    if rng is None:
        rng = random.Random()

    occ = state.occupancy
    n_assigned = len(state.assignments)

    # Compute congestion-adjusted values
    logits = {}
    for f in state.frameworks:
        v = state.base_values[f]
        # Use fractional occupancy based on already-assigned agents
        mu_f = (state.raw_counts.get(f, 0)) / max(n_assigned, 1) if n_assigned > 0 else 0.0
        congestion = gamma * math.log(mu_f + epsilon)
        logits[f] = (v - congestion) / tau

    # Softmax
    max_logit = max(logits.values())
    exp_logits = {f: math.exp(logits[f] - max_logit) for f in state.frameworks}
    total = sum(exp_logits.values())
    probs = {f: exp_logits[f] / total for f in state.frameworks}

    # Sample
    r = rng.random()
    cumulative = 0.0
    for f in state.frameworks:
        cumulative += probs[f]
        if r <= cumulative:
            return f
    return state.frameworks[-1]  # fallback


def assign_frameworks(
    frameworks: list[str],
    n_agents: int,
    gamma: float,
    base_values: dict[str, float] | None = None,
    tau: float = 0.3,
    seed: int | None = None,
) -> tuple[list[str], FrameworkState]:
    """Sequentially assign frameworks to N agents with congestion.

    Returns:
        (assignments, final_state) — list of framework names and the state object.
    """
    rng = random.Random(seed)
    state = FrameworkState(
        frameworks=frameworks,
        base_values=base_values or {f: 1.0 for f in frameworks},
    )

    for _ in range(n_agents):
        chosen = select_framework(state, gamma=gamma, tau=tau, rng=rng)
        state.assignments.append(chosen)

    return state.assignments, state
