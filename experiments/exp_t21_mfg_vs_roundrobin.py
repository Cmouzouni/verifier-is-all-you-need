"""T2.1 — MFG vs Round-Robin head-to-head.

The decisive experiment: does MFG outperform simpler baselines when N≠K?

Setup:
  N = 4 agents
  K = 8 frameworks per task (4 high-value V=1.0, 4 noisy V=0.3)
  3 framework-selection conditions:
    1. MFG (value-aware): P(f) ∝ exp((V_f - γ log μ_f) / τ), γ=1
    2. Random-K: uniform random selection of 4 from 8 (no value information)
    3. Round-robin (fixed): always select the first 4 frameworks in the list

  Topology: Insight+Synth (the cheapest 100% from main sweep)
  Model: qwen-7b throughout (isolate selection effect from model effect)
  Tasks: 30 Phase A tasks with K=8 extended frameworks
  Episodes: 10 per task per condition = 300 episodes per condition

Hypothesis (MFG earns its keep):
  MFG significantly outperforms Random-K (p<0.05, ≥5pp)
  MFG significantly outperforms Round-robin-fixed

Decision rule:
  If MFG passes both, MFG framework stays in the paper.
  Otherwise, drop the MFG framing.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.runner import _extract_answer
from src.config import RESULTS_DIR
from tasks.extended_frameworks import EXTENDED_PHASE_A_TASKS, get_value_priors


# ── Configuration ──────────────────────────────────────────────────────
N_AGENTS = 4
GAMMA = 1.0
TAU = 0.3
TEMPERATURE = 0.7
MAX_TOKENS = 1024
EPISODES_PER_TASK = 10
MODEL_KEY = "qwen-7b"


# ════════════════════════════════════════════════════════════════════════
# THREE FRAMEWORK SELECTION STRATEGIES
# ════════════════════════════════════════════════════════════════════════

def select_mfg(frameworks: list[str], values: dict[str, float],
                n_agents: int, gamma: float, tau: float, rng: random.Random) -> list[str]:
    """MFG value-aware selection with congestion."""
    assignments = []
    counts = {f: 0 for f in frameworks}

    for i in range(n_agents):
        # Compute logits: V_f - γ × log(μ_f + ε)
        logits = {}
        for f in frameworks:
            mu_f = counts[f] / max(i, 1) if i > 0 else 0
            congestion = gamma * math.log(mu_f + 1e-6)
            logits[f] = (values[f] - congestion) / tau

        # Softmax sample
        max_l = max(logits.values())
        exps = {f: math.exp(logits[f] - max_l) for f in frameworks}
        total = sum(exps.values())
        probs = {f: exps[f] / total for f in frameworks}

        r = rng.random()
        cumulative = 0.0
        chosen = frameworks[-1]
        for f in frameworks:
            cumulative += probs[f]
            if r <= cumulative:
                chosen = f
                break

        assignments.append(chosen)
        counts[chosen] += 1

    return assignments


def select_random_k(frameworks: list[str], values: dict[str, float],
                     n_agents: int, gamma: float, tau: float, rng: random.Random) -> list[str]:
    """Random uniform selection of N from K, no value information."""
    return rng.sample(frameworks, n_agents)


def select_round_robin_fixed(frameworks: list[str], values: dict[str, float],
                              n_agents: int, gamma: float, tau: float, rng: random.Random) -> list[str]:
    """Deterministic: always pick the first N frameworks in the list."""
    return list(frameworks[:n_agents])


SELECTORS = {
    "mfg": select_mfg,
    "random_k": select_random_k,
    "round_robin_fixed": select_round_robin_fixed,
}


# ════════════════════════════════════════════════════════════════════════
# EPISODE EXECUTION
# ════════════════════════════════════════════════════════════════════════

def run_propose_pool(client: LLMClient, task, assignments: list[str]) -> list[dict]:
    """Generate one proposal per assigned framework, in parallel."""
    requests = []
    for fw in assignments:
        sys_prompt = task.framework_prompts.get(fw, f"Solve using the {fw} approach.")
        usr_prompt = (
            f"Problem: {task.problem}\n\n"
            f"Solve step by step using the {fw} approach.\n"
            f"State your final answer on a line starting with 'ANSWER: '."
        )
        requests.append((sys_prompt, usr_prompt))
    responses = client.generate_batch(requests, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, max_workers=N_AGENTS)
    return [{
        "framework": fw,
        "answer": _extract_answer(r.content),
        "reasoning": r.content[:1500],
        "cost": r.cost_usd,
        "tokens": r.input_tokens + r.output_tokens,
    } for fw, r in zip(assignments, responses)]


def synthesize(client: LLMClient, task, proposals: list[dict]) -> tuple[str, dict]:
    """Run the LLM synthesizer over the proposals (full reasoning)."""
    text = "\n\n".join(
        f"Agent {i+1} ({p['framework']}):\n{p['reasoning']}"
        for i, p in enumerate(proposals)
    )
    sys_prompt = (
        "You are a senior synthesizer. Multiple agents proposed solutions. "
        "Read each, identify which is correct, and produce the final answer."
    )
    usr_prompt = (
        f"Problem: {task.problem}\n\n"
        f"Agent solutions:\n{text}\n\n"
        f"State the correct final answer on a line starting with 'ANSWER: '."
    )
    resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return _extract_answer(resp.content), {"cost": resp.cost_usd, "tokens": resp.input_tokens + resp.output_tokens}


def run_episode(client: LLMClient, task, condition: str, seed: int) -> dict:
    rng = random.Random(seed)
    values = get_value_priors(task)

    selector = SELECTORS[condition]
    assignments = selector(task.frameworks, values, N_AGENTS, GAMMA, TAU, rng)

    # Generate proposals
    proposals = run_propose_pool(client, task, assignments)

    # Synthesize
    final_answer, synth_meta = synthesize(client, task, proposals)
    correct = task.check(final_answer)

    # Coverage stats
    n_high_value = sum(1 for f in assignments if values[f] >= 0.5)
    n_low_value = sum(1 for f in assignments if values[f] < 0.5)

    proposal_cost = sum(p["cost"] for p in proposals)
    total_cost = proposal_cost + synth_meta["cost"]

    return {
        "task_id": task.id,
        "domain": task.domain,
        "condition": condition,
        "assignments": assignments,
        "n_high_value": n_high_value,
        "n_low_value": n_low_value,
        "proposals": [{"framework": p["framework"], "answer": p["answer"]} for p in proposals],
        "final_answer": final_answer,
        "ground_truth": task.ground_truth,
        "correct": correct,
        "cost": total_cost,
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main(dry_run: bool = False):
    print("=" * 70)
    print("T2.1 — MFG vs ROUND-ROBIN HEAD-TO-HEAD")
    print("=" * 70)
    print(f"  N agents: {N_AGENTS}")
    print(f"  K frameworks: 8 (4 good V=1.0, 4 noisy V=0.3)")
    print(f"  Conditions: {list(SELECTORS.keys())}")
    print(f"  Tasks: {len(EXTENDED_PHASE_A_TASKS)}")
    print(f"  Episodes per task per condition: {EPISODES_PER_TASK}")
    n_eps_per_cond = len(EXTENDED_PHASE_A_TASKS) * EPISODES_PER_TASK
    n_total = n_eps_per_cond * len(SELECTORS)
    print(f"  Total episodes: {n_total} ({n_eps_per_cond} per condition)")

    if dry_run:
        # Each episode: N proposal calls + 1 synth call = N+1 calls
        n_calls = n_total * (N_AGENTS + 1)
        # qwen-7b at $0.30/M, ~800 tokens/call
        est = n_calls * 800 * 0.30 / 1e6
        print(f"\n[DRY RUN] {n_calls} calls, est ${est:.2f}")
        return

    client = LLMClient(MODEL_KEY)
    rng = random.Random(42)
    t_start = time.time()
    output_dir = RESULTS_DIR / "t21_mfg_vs_rr"
    output_dir.mkdir(exist_ok=True)

    all_records = []

    for condition in SELECTORS:
        print(f"\n--- Condition: {condition} ---", flush=True)
        # Build job list
        jobs = []
        for task in EXTENDED_PHASE_A_TASKS:
            for ep in range(EPISODES_PER_TASK):
                jobs.append((task, condition, rng.randint(0, 2**31)))

        cond_records = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(run_episode, client, t, c, s): (t, c) for t, c, s in jobs}
            completed = 0
            for f in as_completed(futures):
                try:
                    cond_records.append(f.result())
                    completed += 1
                    if completed % 50 == 0:
                        k = sum(r["correct"] for r in cond_records)
                        print(f"  [{completed}/{len(jobs)}] {k}/{completed} correct, cost=${client.tracker.total_cost:.4f}", flush=True)
                except Exception as e:
                    print(f"  [ERROR] {e}", flush=True)

        all_records.extend(cond_records)
        k = sum(r["correct"] for r in cond_records)
        n = len(cond_records)
        avg_cost = sum(r["cost"] for r in cond_records) / n
        avg_high = sum(r["n_high_value"] for r in cond_records) / n
        print(f"  → {condition}: {k}/{n} ({k/n:.1%}), avg cost ${avg_cost:.5f}/ep, avg high-value frameworks {avg_high:.2f}/4", flush=True)

    elapsed = time.time() - t_start

    # ── Statistical analysis ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"T2.1 RESULTS")
    print(f"{'='*70}")

    by_cond = defaultdict(list)
    for r in all_records:
        by_cond[r["condition"]].append(r)

    rates = {}
    for cond, recs in by_cond.items():
        k = sum(r["correct"] for r in recs)
        n = len(recs)
        rate = k / n
        # Wilson CI
        z = 1.96
        denom = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denom
        half = z * (rate * (1 - rate) / n + z**2 / (4 * n**2))**0.5 / denom
        lo = max(0, center - half)
        hi = min(1, center + half)
        rates[cond] = {"k": k, "n": n, "rate": rate, "ci_lo": lo, "ci_hi": hi}
        avg_high = sum(r["n_high_value"] for r in recs) / n
        avg_cost = sum(r["cost"] for r in recs) / n
        print(f"  {cond:20s}: {rate:.1%} [{lo:.1%}, {hi:.1%}] (n={n}) | high-V picks {avg_high:.2f}/4 | ${avg_cost:.5f}/ep")

    # Pairwise comparisons
    print(f"\n--- Pairwise comparisons (Fisher exact) ---")

    def fisher_exact(k1, n1, k2, n2):
        """Two-sided Fisher's exact test (basic implementation)."""
        from math import comb
        total = n1 + n2
        k_total = k1 + k2
        observed_diff = abs(k1 / n1 - k2 / n2)
        # Compute via hypergeometric tail
        p_obs = 0
        for k in range(max(0, k_total - n2), min(k_total, n1) + 1):
            if abs(k / n1 - (k_total - k) / n2) >= observed_diff - 1e-9:
                p = comb(n1, k) * comb(n2, k_total - k) / comb(total, k_total)
                p_obs += p
        return min(1.0, p_obs)

    pairs = [("mfg", "random_k"), ("mfg", "round_robin_fixed"), ("random_k", "round_robin_fixed")]
    for a, b in pairs:
        ra, rb = rates[a], rates[b]
        delta = ra["rate"] - rb["rate"]
        try:
            p = fisher_exact(ra["k"], ra["n"], rb["k"], rb["n"])
        except Exception as e:
            p = float("nan")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {a:20s} vs {b:20s}: Δ={delta:+.1%}  p={p:.4f} {sig}")

    # Decision rule
    print(f"\n--- DECISION RULE ---")
    mfg_rate = rates["mfg"]["rate"]
    rr_rate = rates["random_k"]["rate"]
    rrf_rate = rates["round_robin_fixed"]["rate"]
    delta_random = mfg_rate - rr_rate
    delta_rrf = mfg_rate - rrf_rate

    p_vs_random = fisher_exact(rates["mfg"]["k"], rates["mfg"]["n"], rates["random_k"]["k"], rates["random_k"]["n"])
    p_vs_rrf = fisher_exact(rates["mfg"]["k"], rates["mfg"]["n"], rates["round_robin_fixed"]["k"], rates["round_robin_fixed"]["n"])

    print(f"  MFG vs Random-K: Δ={delta_random:+.1%}, p={p_vs_random:.4f}")
    print(f"  MFG vs RR-fixed: Δ={delta_rrf:+.1%}, p={p_vs_rrf:.4f}")

    if delta_random >= 0.05 and p_vs_random < 0.05:
        print(f"\n  ✓ MFG PASSES: significantly outperforms Random-K by ≥5pp")
        print(f"  → Keep MFG framework in the paper. MFG earns its keep when N<K.")
    elif delta_random > 0:
        print(f"\n  ~ MFG PARTIAL: slightly better than Random-K but not significant")
        print(f"  → Borderline. Consider running with more episodes or larger K/N gap.")
    else:
        print(f"\n  ✗ MFG FAILS: not significantly better than Random-K")
        print(f"  → DROP MFG framing from the paper. Rewrite as empirical study.")

    print(f"\n  Total cost: ${client.tracker.total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    json.dump({
        "experiment": "t21_mfg_vs_roundrobin",
        "design": {
            "n_agents": N_AGENTS, "k_frameworks": 8,
            "n_high_value": 4, "n_low_value": 4,
            "high_value": 1.0, "low_value": 0.3,
            "gamma": GAMMA, "tau": TAU,
            "episodes_per_task": EPISODES_PER_TASK,
            "n_tasks": len(EXTENDED_PHASE_A_TASKS),
        },
        "rates": rates,
        "comparisons": {
            "mfg_vs_random_k": {"delta": delta_random, "p_value": p_vs_random},
            "mfg_vs_round_robin_fixed": {"delta": delta_rrf, "p_value": p_vs_rrf},
        },
        "decision": "PASS" if (delta_random >= 0.05 and p_vs_random < 0.05) else ("PARTIAL" if delta_random > 0 else "FAIL"),
        "total_cost_usd": client.tracker.total_cost,
        "elapsed_s": elapsed,
        "episodes": all_records,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
