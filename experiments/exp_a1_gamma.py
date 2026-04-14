"""Experiment A.1 — γ Response Validation

Question: Do agents actually diversify behavior when γ increases?

Design:
  - 30 simple multi-approach tasks (not benchmarks)
  - Model: Qwen2.5-3B exclusively (isolate γ response from capability)
  - Graph: Propose state only, population-coupled softmax selection
  - N = 4 agents
  - γ sweep: {0, 0.5, 1, 2, 4, 8}
  - Episodes: 20 per γ value (tasks sampled from pool)

Primary metric: D(γ) = 1 - max_i(µ_i), should be monotonically increasing.
Statistical test: Jonckheere-Terpstra for monotonic trend.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.runner import run_propose_episode
from src.diversity import diversity_score, framework_coverage, shannon_entropy, mfg_predicted_diversity
from src.logger import ExperimentLogger, EpisodeLog
from src.config import RESULTS_DIR
from tasks.phase_a_tasks import PHASE_A_TASKS


# ── Configuration ──────────────────────────────────────────────────────
EXPERIMENT_ID = "exp_a1_gamma"
MODEL_KEY = "qwen-7b"  # smallest Qwen tier — isolate γ response from capability
N_AGENTS = 4
GAMMA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
EPISODES_PER_GAMMA = 20
TAU = 0.3
TEMPERATURE = 0.7
MAX_TOKENS = 512


def run_experiment(dry_run: bool = False) -> dict:
    """Run the full A.1 γ sweep experiment."""
    print(f"{'='*60}")
    print(f"Experiment A.1: γ Response Validation")
    print(f"Model: {MODEL_KEY} | N={N_AGENTS} | Episodes/γ={EPISODES_PER_GAMMA}")
    print(f"γ values: {GAMMA_VALUES}")
    print(f"{'='*60}")

    if dry_run:
        total_calls = len(GAMMA_VALUES) * EPISODES_PER_GAMMA * N_AGENTS
        est_cost = total_calls * 800 * 0.08 / 1_000_000  # ~800 tokens/call, ~$0.08/M avg
        print(f"\n[DRY RUN] Would make {total_calls} API calls")
        print(f"[DRY RUN] Estimated cost: ${est_cost:.3f}")
        print(f"[DRY RUN] Estimated time: {total_calls * 1.5 / 60:.1f} minutes")
        return {}

    client = LLMClient(model_key=MODEL_KEY)
    logger = ExperimentLogger(EXPERIMENT_ID, RESULTS_DIR / EXPERIMENT_ID)
    rng = random.Random(42)

    all_results = {}
    t_start = time.time()

    for gamma in GAMMA_VALUES:
        print(f"\n--- γ = {gamma} ---")
        gamma_results = []

        # Sample tasks for this γ level
        tasks = rng.choices(PHASE_A_TASKS, k=EPISODES_PER_GAMMA)

        for ep_idx, task in enumerate(tasks):
            seed = rng.randint(0, 2**31)

            result = run_propose_episode(
                client=client,
                task=task,
                n_agents=N_AGENTS,
                gamma=gamma,
                tau=TAU,
                seed=seed,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            gamma_results.append(result)

            # Build episode log
            ep_log = EpisodeLog(
                experiment_id=EXPERIMENT_ID,
                episode_id=f"ep_{gamma}_{ep_idx:03d}",
                task_id=task.id,
                benchmark="phase_a",
                difficulty=task.difficulty,
                gamma=gamma,
                n_agents=N_AGENTS,
                frameworks_available=task.frameworks,
                framework_assignments=result.assignments,
                diversity_score=result.diversity,
                framework_coverage=result.coverage,
                shannon_entropy=result.entropy,
                population_distribution=result.population_dist,
                agent_logs=result.agent_logs,
                ground_truth=task.ground_truth,
                solutions=result.solutions,
                correct_count=sum(result.correct),
                total_tokens=result.total_tokens,
                total_cost_usd=result.total_cost,
                wall_time_s=result.wall_time,
            )
            logger.add_episode(ep_log)

            if (ep_idx + 1) % 5 == 0:
                mean_d = sum(r.diversity for r in gamma_results) / len(gamma_results)
                mean_correct = sum(sum(r.correct) for r in gamma_results) / (len(gamma_results) * N_AGENTS)
                print(f"  [{ep_idx+1}/{EPISODES_PER_GAMMA}] D={mean_d:.3f} correct={mean_correct:.1%} cost=${client.tracker.total_cost:.4f}")

        # Summarize γ level
        diversities = [r.diversity for r in gamma_results]
        coverages = [r.coverage for r in gamma_results]
        mean_d = sum(diversities) / len(diversities)
        mean_cov = sum(coverages) / len(coverages)
        all_results[gamma] = {
            "diversities": diversities,
            "coverages": coverages,
            "mean_diversity": mean_d,
            "mean_coverage": mean_cov,
            "correct_rate": sum(sum(r.correct) for r in gamma_results) / (len(gamma_results) * N_AGENTS),
        }
        print(f"  γ={gamma}: mean D={mean_d:.3f}, coverage={mean_cov:.3f}, "
              f"correct={all_results[gamma]['correct_rate']:.1%}")

    # ── Save results ───────────────────────────────────────────────────
    path = logger.save()
    print(f"\nResults saved to: {path}")
    print(f"Total cost: ${client.tracker.total_cost:.4f}")
    print(f"Total tokens: {client.tracker.total_tokens:,}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    # ── Compute MFG predictions for comparison ─────────────────────────
    print(f"\n--- MFG Predicted vs Observed D(γ) ---")
    print(f"{'γ':>6} {'Predicted':>10} {'Observed':>10} {'|Δ|':>8}")
    l1_total = 0.0
    for gamma in GAMMA_VALUES:
        predicted = mfg_predicted_diversity(gamma, N_AGENTS, 4, TAU)
        observed = all_results[gamma]["mean_diversity"]
        delta = abs(predicted - observed)
        l1_total += delta
        print(f"{gamma:>6.1f} {predicted:>10.3f} {observed:>10.3f} {delta:>8.3f}")
    l1_mean = l1_total / len(GAMMA_VALUES)
    print(f"\nMean L1 distance: {l1_mean:.4f}")
    print(f"L1 target: < 0.15 (tight), < 0.30 (acceptable)")

    # ── Statistical test ───────────────────────────────────────────────
    print(f"\n--- Monotonic Trend Test ---")
    d_values_per_gamma = [all_results[g]["diversities"] for g in GAMMA_VALUES]
    jt_stat, jt_p = jonckheere_terpstra(d_values_per_gamma)
    print(f"Jonckheere-Terpstra statistic: {jt_stat:.2f}")
    print(f"p-value: {jt_p:.6f}")
    print(f"Monotonic trend {'CONFIRMED' if jt_p < 0.05 else 'NOT CONFIRMED'} (p < 0.05)")

    # ── Gate evaluation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GO/NO-GO GATE A.1:")
    go = jt_p < 0.05 and l1_mean < 0.30
    if jt_p < 0.05 and l1_mean < 0.15:
        print("  ✓ STRONG GO: Monotonic trend confirmed, tight MFG fit")
    elif go:
        print("  ✓ GO: Monotonic trend confirmed, acceptable MFG fit")
    elif jt_p < 0.05:
        print("  ~ PARTIAL GO: Trend confirmed but MFG fit loose (L1 > 0.30)")
    else:
        print("  ✗ NO-GO: No monotonic trend detected")
    print(f"{'='*60}")

    # Save analysis summary
    summary = {
        "experiment": EXPERIMENT_ID,
        "model": MODEL_KEY,
        "n_agents": N_AGENTS,
        "gamma_values": GAMMA_VALUES,
        "episodes_per_gamma": EPISODES_PER_GAMMA,
        "results": {str(g): all_results[g] for g in GAMMA_VALUES},
        "l1_mean": l1_mean,
        "jt_statistic": jt_stat,
        "jt_p_value": jt_p,
        "gate": "GO" if go else "NO-GO",
        "total_cost_usd": client.tracker.total_cost,
        "total_tokens": client.tracker.total_tokens,
    }
    summary_path = RESULTS_DIR / EXPERIMENT_ID / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return all_results


def jonckheere_terpstra(groups: list[list[float]]) -> tuple[float, float]:
    """Jonckheere-Terpstra test for ordered alternatives (monotonic trend).

    H0: No monotonic trend across ordered groups.
    H1: Values increase with group index.

    Returns (statistic, p_value).
    """
    import math

    k = len(groups)
    # Count concordant pairs
    U = 0
    total_pairs = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        U += 1
                    elif xj == xi:
                        U += 0.5
                    total_pairs += 1

    # Expected value and variance under H0
    ns = [len(g) for g in groups]
    N = sum(ns)
    E_U = (N * N - sum(n * n for n in ns)) / 4

    # Variance (without ties correction for simplicity)
    num = N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ns)
    var_U = num / 72

    if var_U <= 0:
        return 0.0, 1.0

    Z = (U - E_U) / math.sqrt(var_U)

    # One-sided p-value (testing for increasing trend)
    # Use normal approximation
    p = 0.5 * (1 - _erf(Z / math.sqrt(2)))

    return Z, p


def _erf(x: float) -> float:
    """Approximate error function."""
    import math
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t * math.exp(-x * x)
    return sign * y


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_experiment(dry_run=dry_run)
