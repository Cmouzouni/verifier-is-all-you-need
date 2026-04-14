"""Tier 1.2 — Synthesis ablation.

Question: Is the synthesis effect from reading reasoning content, or just from
having any LLM-level aggregation?

Three aggregator conditions:
  A. Majority vote (no synthesis, current Propose-only baseline)
  B. LLM synthesizer reading ONLY final answers (not reasoning)
  C. LLM synthesizer reading FULL reasoning traces (current Insight+Synth)

Holding constant: 4 × qwen-7b agents using insight roles, γ=1.

If B ≈ A: synthesis effect is about reading reasoning → reasoning matters.
If B ≈ C: synthesis effect is about LLM aggregation → reasoning doesn't matter.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.congestion import assign_frameworks
from src.runner import _extract_answer
from src.config import RESULTS_DIR
from tasks.phase_a_tasks import PHASE_A_TASKS
from tasks.gsm8k_tasks import load_gsm8k


# ── Configuration ──────────────────────────────────────────────────────
N_AGENTS = 4
GAMMA = 1.0
TAU = 0.3
TEMPERATURE = 0.7
MAX_TOKENS = 1024
EPISODES_PER_TASK = 5  # for statistical power


def generate_proposals(client, task, gamma, seed):
    """Generate N independent proposals using framework congestion."""
    assignments, _ = assign_frameworks(
        frameworks=task.frameworks,
        n_agents=N_AGENTS,
        gamma=gamma, tau=TAU, seed=seed,
    )
    proposals = []
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
    for resp, fw in zip(responses, assignments):
        answer = _extract_answer(resp.content)
        proposals.append({
            "framework": fw,
            "reasoning": resp.content[:1500],
            "answer": answer,
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        })
    return proposals, assignments


def aggregate_majority_vote(proposals: list) -> str:
    """Plain majority vote over the answers."""
    answers = [p["answer"] for p in proposals if p["answer"]]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def aggregate_synth_answers_only(client, task, proposals: list) -> tuple[str, dict]:
    """LLM synthesizer that sees ONLY the answers (not the reasoning)."""
    answers_text = "\n".join(
        f"Agent {i+1} ({p['framework']}): {p['answer']}"
        for i, p in enumerate(proposals)
    )
    sys_prompt = (
        "You are a senior synthesizer. Multiple agents proposed answers to the problem. "
        "You see only their final answers, not their reasoning. "
        "Pick the correct answer."
    )
    usr_prompt = (
        f"Problem: {task.problem}\n\n"
        f"Agent answers:\n{answers_text}\n\n"
        f"State the correct final answer on a line starting with 'ANSWER: '."
    )
    resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return _extract_answer(resp.content), {"tokens": resp.input_tokens + resp.output_tokens, "cost": resp.cost_usd}


def aggregate_synth_full_reasoning(client, task, proposals: list) -> tuple[str, dict]:
    """LLM synthesizer that sees FULL reasoning traces (current default)."""
    full_text = "\n\n".join(
        f"Agent {i+1} ({p['framework']}):\n{p['reasoning']}"
        for i, p in enumerate(proposals)
    )
    sys_prompt = (
        "You are a senior synthesizer. Multiple agents proposed solutions to the problem. "
        "Read each agent's full reasoning, identify which is correct, and produce the final answer."
    )
    usr_prompt = (
        f"Problem: {task.problem}\n\n"
        f"Agent solutions:\n{full_text}\n\n"
        f"State the correct final answer on a line starting with 'ANSWER: '."
    )
    resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return _extract_answer(resp.content), {"tokens": resp.input_tokens + resp.output_tokens, "cost": resp.cost_usd}


def run_episode(client, task, seed) -> dict:
    """One episode with all 3 aggregator conditions on the SAME proposals."""
    proposals, assignments = generate_proposals(client, task, GAMMA, seed)

    # Condition A: majority vote
    answer_a = aggregate_majority_vote(proposals)

    # Condition B: synth answers only
    answer_b, meta_b = aggregate_synth_answers_only(client, task, proposals)

    # Condition C: synth full reasoning
    answer_c, meta_c = aggregate_synth_full_reasoning(client, task, proposals)

    proposal_cost = sum(p["cost"] for p in proposals)
    proposal_tokens = sum(p["tokens"] for p in proposals)

    return {
        "task_id": task.id,
        "ground_truth": task.ground_truth,
        "frameworks_assigned": assignments,
        "proposal_cost": proposal_cost,
        "proposal_tokens": proposal_tokens,
        "majority_vote": {
            "answer": answer_a,
            "correct": task.check(answer_a),
            "extra_cost": 0,
        },
        "synth_answers_only": {
            "answer": answer_b,
            "correct": task.check(answer_b),
            "extra_cost": meta_b["cost"],
        },
        "synth_full_reasoning": {
            "answer": answer_c,
            "correct": task.check(answer_c),
            "extra_cost": meta_c["cost"],
        },
    }


def main(dry_run: bool = False):
    print("=" * 70)
    print("TIER 1.2 — SYNTHESIS ABLATION")
    print("=" * 70)

    # Use BOTH Phase A tasks AND GSM8K subset for breadth
    phase_a = PHASE_A_TASKS
    gsm8k = load_gsm8k(n=20, seed=42)
    all_tasks = phase_a + gsm8k
    print(f"Tasks: {len(phase_a)} Phase A + {len(gsm8k)} GSM8K = {len(all_tasks)}")
    print(f"Episodes per task: {EPISODES_PER_TASK}")
    print(f"Total episodes: {len(all_tasks) * EPISODES_PER_TASK}")

    if dry_run:
        # Each episode: 4 proposals + 2 synth calls = 6 calls
        n_calls = len(all_tasks) * EPISODES_PER_TASK * 6
        est_cost = n_calls * 800 * 0.30 / 1e6
        print(f"\n[DRY RUN] Estimated cost: ${est_cost:.2f}")
        print(f"[DRY RUN] API calls: {n_calls}")
        return

    client = LLMClient("qwen-7b")
    rng = random.Random(42)
    t_start = time.time()

    # Run episodes in parallel batches
    jobs = []
    for task in all_tasks:
        for ep in range(EPISODES_PER_TASK):
            jobs.append((task, rng.randint(0, 2**31)))

    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(run_episode, client, task, seed): (task, seed) for task, seed in jobs}
        completed = 0
        for f in as_completed(futures):
            try:
                results.append(f.result())
                completed += 1
                if completed % 25 == 0:
                    print(f"  [{completed}/{len(jobs)}] cost so far: ${client.tracker.total_cost:.4f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)

    elapsed = time.time() - t_start

    # ── Aggregate ──────────────────────────────────────────────────────
    n = len(results)
    rate_a = sum(r["majority_vote"]["correct"] for r in results) / n
    rate_b = sum(r["synth_answers_only"]["correct"] for r in results) / n
    rate_c = sum(r["synth_full_reasoning"]["correct"] for r in results) / n

    cost_a = sum(r["proposal_cost"] for r in results) / n
    cost_b = (sum(r["proposal_cost"] for r in results) + sum(r["synth_answers_only"]["extra_cost"] for r in results)) / n
    cost_c = (sum(r["proposal_cost"] for r in results) + sum(r["synth_full_reasoning"]["extra_cost"] for r in results)) / n

    print(f"\n{'='*70}")
    print(f"TIER 1.2 RESULTS — Synthesis Ablation")
    print(f"{'='*70}")
    print(f"\n  Episodes: {n}")
    print(f"\n  A. Majority vote (no synth):       {rate_a:.1%} cost=${cost_a:.5f}")
    print(f"  B. Synth (answers only):           {rate_b:.1%} cost=${cost_b:.5f}")
    print(f"  C. Synth (full reasoning):         {rate_c:.1%} cost=${cost_c:.5f}")

    if rate_b >= rate_a + 0.05 and rate_c >= rate_b + 0.05:
        print(f"\n  → Reading FULL REASONING is the key (C > B > A)")
    elif rate_c >= rate_a + 0.05 and abs(rate_b - rate_c) < 0.05:
        print(f"\n  → LLM AGGREGATION is the key (B ≈ C >> A)")
    elif rate_b >= rate_a + 0.05 and abs(rate_c - rate_b) < 0.05:
        print(f"\n  → Even ANSWERS-ONLY synth helps (B >> A, C ≈ B)")
    else:
        print(f"\n  → Mixed result, see numbers")

    print(f"\n  Total cost: ${client.tracker.total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s")

    # Save
    output_dir = RESULTS_DIR / "tier1_synthesis"
    output_dir.mkdir(exist_ok=True)
    json.dump({
        "experiment": "tier1_synthesis_ablation",
        "n_episodes": n,
        "rates": {"majority_vote": rate_a, "synth_answers_only": rate_b, "synth_full_reasoning": rate_c},
        "costs": {"majority_vote": cost_a, "synth_answers_only": cost_b, "synth_full_reasoning": cost_c},
        "total_cost_usd": client.tracker.total_cost,
        "elapsed_s": elapsed,
        "episodes": results,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
