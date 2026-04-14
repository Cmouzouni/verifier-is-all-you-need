"""T2.1-bis — MFG vs Round-Robin in the N >>> K regime.

The original T2.1 tested N=4, K=8 (selection regime). MFG showed no advantage.

This experiment tests N=32, K=8 (allocation regime). With more agents than
frameworks, all three strategies can cover all frameworks; what differs is
HOW they allocate agents across frameworks:
  - Round-robin: exactly N/K = 4 agents per framework (perfectly uniform)
  - Random: N/K agents per framework on average, with variance
  - MFG: more agents on high-value frameworks, fewer on low-value (biased)

The hypothesis: in this regime, MFG's biased allocation toward high-value
frameworks should produce a measurable lift over uniform allocation.

If MFG passes here, the framework earns its keep in the N >> K regime, and
the paper's claim becomes "MFG matters when N exceeds K with informative priors."
If MFG still fails here, drop the framework definitively.
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
N_AGENTS = 32  # KEY CHANGE: 32 agents
GAMMA = 1.0
TAU = 0.3
TEMPERATURE = 0.7
MAX_TOKENS = 1024
EPISODES_PER_TASK = 5  # 30 tasks × 5 episodes = 150 per condition
MODEL_KEY = "qwen-7b"
WORKERS = 30


# ════════════════════════════════════════════════════════════════════════
# THREE FRAMEWORK SELECTION STRATEGIES (N >> K version)
# ════════════════════════════════════════════════════════════════════════

def select_mfg_n32(frameworks: list[str], values: dict[str, float],
                    n_agents: int, gamma: float, tau: float, rng: random.Random) -> list[str]:
    """MFG value-aware selection with congestion. N can be > K."""
    assignments = []
    counts = {f: 0 for f in frameworks}

    for i in range(n_agents):
        logits = {}
        for f in frameworks:
            mu_f = counts[f] / max(i, 1) if i > 0 else 0
            congestion = gamma * math.log(mu_f + 1e-6)
            logits[f] = (values[f] - congestion) / tau

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


def select_random_uniform(frameworks: list[str], values: dict[str, float],
                            n_agents: int, gamma: float, tau: float, rng: random.Random) -> list[str]:
    """Each agent picks uniformly at random from K frameworks. No value info."""
    return [rng.choice(frameworks) for _ in range(n_agents)]


def select_round_robin_balanced(frameworks: list[str], values: dict[str, float],
                                  n_agents: int, gamma: float, tau: float, rng: random.Random) -> list[str]:
    """Deterministic round-robin: exactly N/K agents per framework."""
    return [frameworks[i % len(frameworks)] for i in range(n_agents)]


SELECTORS = {
    "mfg": select_mfg_n32,
    "random_uniform": select_random_uniform,
    "round_robin_balanced": select_round_robin_balanced,
}


# ════════════════════════════════════════════════════════════════════════
# EPISODE EXECUTION
# ════════════════════════════════════════════════════════════════════════

def run_propose_pool(client: LLMClient, task, assignments: list[str]) -> list[dict]:
    """Generate one proposal per assignment, in parallel batches."""
    requests = []
    for fw in assignments:
        sys_prompt = task.framework_prompts.get(fw, f"Solve using the {fw} approach.")
        usr_prompt = (
            f"Problem: {task.problem}\n\n"
            f"Solve step by step using the {fw} approach.\n"
            f"State your final answer on a line starting with 'ANSWER: '."
        )
        requests.append((sys_prompt, usr_prompt))
    responses = client.generate_batch(requests, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, max_workers=16)
    return [{
        "framework": fw,
        "answer": _extract_answer(r.content),
        "reasoning": r.content[:1500],
        "cost": r.cost_usd,
    } for fw, r in zip(assignments, responses)]


def synthesize(client: LLMClient, task, proposals: list[dict]) -> tuple[str, dict]:
    """LLM synth over the (potentially many) proposals.

    With N=32 proposals, the prompt would be huge. Take top-K by quality?
    Simpler: cluster by exact answer match, present unique answers with their counts.
    """
    # Group proposals by canonical answer
    answer_groups = defaultdict(list)
    for p in proposals:
        ans = p["answer"].strip().lower()
        answer_groups[ans].append(p)

    # Build a compact summary: each unique answer with its count and one reasoning sample
    summary_lines = []
    for ans, group in sorted(answer_groups.items(), key=lambda x: -len(x[1])):
        if not ans:
            continue
        count = len(group)
        sample = group[0]
        summary_lines.append(
            f"--- Answer '{ans}' (proposed by {count} agents) ---\n"
            f"Sample reasoning ({sample['framework']}): {sample['reasoning'][:600]}"
        )
    summary_text = "\n\n".join(summary_lines[:8])  # cap at 8 unique answers

    sys_prompt = (
        "You are a senior synthesizer. Multiple agents proposed answers to the problem; "
        "they have been grouped by unique answer with the count of agents who voted for each. "
        "Read the reasoning samples and pick the correct final answer."
    )
    usr_prompt = (
        f"Problem: {task.problem}\n\n"
        f"Agent answer groups:\n{summary_text}\n\n"
        f"State the correct final answer on a line starting with 'ANSWER: '."
    )
    resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return _extract_answer(resp.content), {"cost": resp.cost_usd}


def run_episode(client: LLMClient, task, condition: str, seed: int) -> dict:
    rng = random.Random(seed)
    values = get_value_priors(task)

    selector = SELECTORS[condition]
    assignments = selector(task.frameworks, values, N_AGENTS, GAMMA, TAU, rng)

    proposals = run_propose_pool(client, task, assignments)

    final_answer, synth_meta = synthesize(client, task, proposals)
    correct = task.check(final_answer)

    n_high_value = sum(1 for f in assignments if values[f] >= 0.5)
    proposal_cost = sum(p["cost"] for p in proposals)
    total_cost = proposal_cost + synth_meta["cost"]

    return {
        "task_id": task.id,
        "condition": condition,
        "n_agents": N_AGENTS,
        "assignments_summary": dict(Counter(assignments)),
        "n_high_value": n_high_value,
        "n_low_value": N_AGENTS - n_high_value,
        "any_proposer_correct": any(task.check(p["answer"]) for p in proposals),
        "n_proposers_correct": sum(1 for p in proposals if task.check(p["answer"])),
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
    print("T2.1-bis — MFG vs ROUND-ROBIN in N >>> K regime")
    print("=" * 70)
    print(f"  N agents: {N_AGENTS} (vs original T2.1 with N=4)")
    print(f"  K frameworks: 8 (4 high-value V=1.0, 4 noisy V=0.3)")
    print(f"  Conditions: {list(SELECTORS.keys())}")
    print(f"  Tasks: {len(EXTENDED_PHASE_A_TASKS)}")
    print(f"  Episodes per task per condition: {EPISODES_PER_TASK}")
    n_eps_per_cond = len(EXTENDED_PHASE_A_TASKS) * EPISODES_PER_TASK
    n_total = n_eps_per_cond * len(SELECTORS)
    print(f"  Total episodes: {n_total} ({n_eps_per_cond} per condition)")

    if dry_run:
        # Each episode: 32 proposal calls + 1 synth call = 33 calls
        n_calls = n_total * (N_AGENTS + 1)
        est = n_calls * 800 * 0.30 / 1e6
        print(f"\n[DRY RUN] {n_calls} calls, est ${est:.2f}")
        return

    client = LLMClient(MODEL_KEY)
    rng = random.Random(42)
    t_start = time.time()
    output_dir = RESULTS_DIR / "t21_n32"
    output_dir.mkdir(exist_ok=True)

    all_records = []

    for condition in SELECTORS:
        print(f"\n--- Condition: {condition} ---", flush=True)
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
                    if completed % 30 == 0:
                        k = sum(r["correct"] for r in cond_records)
                        any_k = sum(r["any_proposer_correct"] for r in cond_records)
                        print(f"  [{completed}/{len(jobs)}] synth={k}/{completed} any-prop={any_k}/{completed} cost=${client.tracker.total_cost:.4f}", flush=True)
                except Exception as e:
                    print(f"  [ERROR] {e}", flush=True)

        all_records.extend(cond_records)
        k = sum(r["correct"] for r in cond_records)
        n = len(cond_records)
        avg_high = sum(r["n_high_value"] for r in cond_records) / n
        avg_props_correct = sum(r["n_proposers_correct"] for r in cond_records) / n
        print(f"  → {condition}: {k}/{n} ({k/n:.1%}), avg high-V agents {avg_high:.1f}/{N_AGENTS}, "
              f"avg props correct {avg_props_correct:.1f}/{N_AGENTS}", flush=True)

    elapsed = time.time() - t_start

    # ── Statistical analysis ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"T2.1-bis RESULTS (N=32, K=8)")
    print(f"{'='*70}")

    by_cond = defaultdict(list)
    for r in all_records:
        by_cond[r["condition"]].append(r)

    rates = {}
    for cond, recs in by_cond.items():
        k = sum(r["correct"] for r in recs)
        n = len(recs)
        rate = k / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denom
        half = z * (rate * (1 - rate) / n + z**2 / (4 * n**2))**0.5 / denom
        lo, hi = max(0, center - half), min(1, center + half)
        rates[cond] = {"k": k, "n": n, "rate": rate, "ci_lo": lo, "ci_hi": hi}
        avg_high = sum(r["n_high_value"] for r in recs) / n
        avg_props = sum(r["n_proposers_correct"] for r in recs) / n
        print(f"  {cond:25s}: {rate:.1%} [{lo:.1%}, {hi:.1%}] (n={n}) | "
              f"high-V {avg_high:.1f}/{N_AGENTS} | props correct {avg_props:.1f}/{N_AGENTS}")

    # Pairwise comparisons
    def fisher_exact(k1, n1, k2, n2):
        from math import comb
        total = n1 + n2
        k_total = k1 + k2
        observed_diff = abs(k1 / n1 - k2 / n2)
        p_obs = 0
        for k in range(max(0, k_total - n2), min(k_total, n1) + 1):
            if abs(k / n1 - (k_total - k) / n2) >= observed_diff - 1e-9:
                p = comb(n1, k) * comb(n2, k_total - k) / comb(total, k_total)
                p_obs += p
        return min(1.0, p_obs)

    print(f"\n--- Pairwise comparisons ---")
    pairs = [("mfg", "random_uniform"), ("mfg", "round_robin_balanced"), ("random_uniform", "round_robin_balanced")]
    for a, b in pairs:
        ra, rb = rates[a], rates[b]
        delta = ra["rate"] - rb["rate"]
        try:
            p = fisher_exact(ra["k"], ra["n"], rb["k"], rb["n"])
        except Exception:
            p = float("nan")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {a:25s} vs {b:25s}: Δ={delta:+.1%}  p={p:.4f} {sig}")

    # Decision rule
    print(f"\n--- DECISION RULE ---")
    mfg_rate = rates["mfg"]["rate"]
    rr_rate = rates["round_robin_balanced"]["rate"]
    rand_rate = rates["random_uniform"]["rate"]
    delta = mfg_rate - rr_rate
    p_vs_rr = fisher_exact(rates["mfg"]["k"], rates["mfg"]["n"],
                            rates["round_robin_balanced"]["k"], rates["round_robin_balanced"]["n"])

    print(f"  MFG vs Round-robin balanced: Δ={delta:+.1%}, p={p_vs_rr:.4f}")
    if delta >= 0.03 and p_vs_rr < 0.05:
        print(f"\n  ✓✓✓ MFG PASSES IN N >>> K REGIME")
        print(f"  → Keep MFG framework. The framework earns its keep when N exceeds K with informative priors.")
    elif delta > 0:
        print(f"\n  ~ MFG slightly better but not significant")
        print(f"  → Borderline. Need more episodes or larger N.")
    else:
        print(f"\n  ✗ MFG STILL FAILS IN N >>> K REGIME")
        print(f"  → Drop MFG definitively. The framework provides no value at any N tested.")

    print(f"\n  Total cost: ${client.tracker.total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    json.dump({
        "experiment": "t21_n32_mfg_vs_rr",
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
            "mfg_vs_random": {"delta": mfg_rate - rand_rate, "p_value": fisher_exact(rates["mfg"]["k"], rates["mfg"]["n"], rates["random_uniform"]["k"], rates["random_uniform"]["n"])},
            "mfg_vs_round_robin": {"delta": delta, "p_value": p_vs_rr},
        },
        "decision": "PASS" if (delta >= 0.03 and p_vs_rr < 0.05) else ("PARTIAL" if delta > 0 else "FAIL"),
        "total_cost_usd": client.tracker.total_cost,
        "elapsed_s": elapsed,
        "episodes": all_records,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
