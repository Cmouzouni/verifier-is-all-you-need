"""Tier 1.3 — Model family replication.

Question: Does the synthesis-completed topology effect generalize beyond Qwen?

Setup: Replicate the cheapest 100% configuration on two non-Qwen model families
that we verified are available on Together AI:
  - Llama-3.3-70B (only Llama serverless model available)
  - DeepSeek-V3.1 (frontier non-Qwen open model)

We can't replicate the heterogeneous-cheap configuration exactly because no
small Llama (<8B) is serverless. So we test:
  1. all-llama70b × insight-synth × γ=1
  2. all-deepseek × insight-synth × γ=1

Compared against:
  - Single Llama-3.3-70B baseline
  - Single DeepSeek-V3.1 baseline
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.runner import _extract_answer
from src.config import RESULTS_DIR
from experiments.exp_full_sweep_parallel import run_episode_parallel
from experiments.exp_full_sweep import TOPOLOGIES
from tasks.phase_a_tasks import PHASE_A_TASKS


# ── Configuration ──────────────────────────────────────────────────────
EPISODES_PER_TASK = 5
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Replication configurations: same recipe, different model family
REPLICATION_CONFIGS = [
    ("all-llama70b", {"propose": "llama-70b", "critique": "llama-70b", "synthesize": "llama-70b"}),
    ("all-deepseek", {"propose": "deepseek", "critique": "deepseek", "synthesize": "deepseek"}),
]

# Test the cheapest 100% topology recipe
TOPOLOGY_NAME = "all-insight-synth"
GAMMA = 1.0


def run_baseline(client: LLMClient, task) -> dict:
    sys_prompt = "You are a problem solver. Solve step by step. State your final answer on a line starting with 'ANSWER: '."
    usr_prompt = f"Problem: {task.problem}"
    try:
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        answer = _extract_answer(resp.content)
        return {
            "task_id": task.id,
            "answer": answer,
            "correct": task.check(answer),
            "cost": resp.cost_usd,
            "tokens": resp.input_tokens + resp.output_tokens,
        }
    except Exception as e:
        return {"task_id": task.id, "answer": "", "correct": False, "cost": 0, "tokens": 0, "error": str(e)[:80]}


def main(dry_run: bool = False):
    print("=" * 70)
    print("TIER 1.3 — MODEL FAMILY REPLICATION")
    print("=" * 70)

    tasks = PHASE_A_TASKS
    print(f"Tasks: {len(tasks)} (Phase A)")
    print(f"Episodes per task: {EPISODES_PER_TASK}")
    print(f"Topology: {TOPOLOGY_NAME} @ γ={GAMMA}")
    print(f"Configs: {[c[0] for c in REPLICATION_CONFIGS]}")

    if dry_run:
        n_baseline = len(REPLICATION_CONFIGS) * len(tasks)
        n_game = len(REPLICATION_CONFIGS) * len(tasks) * EPISODES_PER_TASK
        n_calls = n_baseline + n_game * 4  # 4 agents per game episode
        # Llama-70B and DeepSeek prices
        avg_cost = 0.9 * 800 / 1e6
        est = n_calls * avg_cost
        print(f"\n[DRY RUN] {n_calls} calls, est ${est:.2f}")
        return

    clients = {
        "llama-70b": LLMClient("llama-70b"),
        "deepseek": LLMClient("deepseek"),
    }

    t_start = time.time()
    output_dir = RESULTS_DIR / "tier1_family"
    output_dir.mkdir(exist_ok=True)

    results = {"baselines": {}, "game": {}}

    # ── Baselines ──────────────────────────────────────────────────────
    print("\n--- BASELINES ---")
    for model_key in ["llama-70b", "deepseek"]:
        client = clients[model_key]
        print(f"  {model_key} ({len(tasks)} tasks)...", flush=True)
        baseline_recs = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(run_baseline, client, t) for t in tasks]
            for f in as_completed(futures):
                baseline_recs.append(f.result())
        k = sum(r["correct"] for r in baseline_recs)
        n = len(baseline_recs)
        cost = sum(r["cost"] for r in baseline_recs)
        results["baselines"][model_key] = {
            "rate": k / n,
            "k": k, "n": n, "total_cost": cost,
            "records": baseline_recs,
        }
        print(f"    {model_key}: {k}/{n} ({k/n:.1%}) — ${cost:.4f}", flush=True)

    # ── Game configurations ────────────────────────────────────────────
    print("\n--- GAME CONFIGS ---")
    rng = random.Random(42)
    tp = TOPOLOGIES[TOPOLOGY_NAME]

    for config_name, config_dict in REPLICATION_CONFIGS:
        print(f"\n  {config_name} × {TOPOLOGY_NAME} γ={GAMMA}", flush=True)

        # Build job list
        jobs = []
        for task in tasks:
            for ep in range(EPISODES_PER_TASK):
                jobs.append((task, rng.randint(0, 2**31)))

        records = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(
                    run_episode_parallel,
                    task=task, gamma=GAMMA,
                    model_config_name=config_name, model_config=config_dict,
                    topology_name=TOPOLOGY_NAME, topology=tp,
                    clients=clients, seed=seed,
                ): task
                for task, seed in jobs
            }
            for f in as_completed(futures):
                try:
                    records.append(f.result())
                except Exception as e:
                    print(f"    [ERROR] {e}", flush=True)

        if records:
            k = sum(r.correct for r in records)
            n = len(records)
            cost = sum(r.total_cost_usd for r in records) / n
            print(f"    -> {k}/{n} ({k/n:.1%}) avg_cost/ep=${cost:.5f}", flush=True)
            results["game"][config_name] = {
                "rate": k / n, "k": k, "n": n, "avg_cost": cost,
                "records": [asdict(r) for r in records],
            }

    elapsed = time.time() - t_start
    total_cost = sum(c.tracker.total_cost for c in clients.values())

    # ── Save and report ────────────────────────────────────────────────
    json.dump({
        "experiment": "tier1_family_replication",
        "topology": TOPOLOGY_NAME,
        "gamma": GAMMA,
        "n_tasks": len(tasks),
        "episodes_per_task": EPISODES_PER_TASK,
        "results": results,
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"TIER 1.3 RESULTS — Family Replication")
    print(f"{'='*70}")

    for model_key in ["llama-70b", "deepseek"]:
        b = results["baselines"][model_key]
        g_key = f"all-{model_key.replace('-', '')}"
        if g_key not in results["game"]:
            g_key = "all-llama70b" if model_key == "llama-70b" else "all-deepseek"
        if g_key in results["game"]:
            g = results["game"][g_key]
            print(f"\n  {model_key}:")
            print(f"    Single baseline: {b['rate']:.1%} ({b['k']}/{b['n']}) — ${b['total_cost']/b['n']:.5f}/prob")
            print(f"    Game ({TOPOLOGY_NAME}, γ={GAMMA}): {g['rate']:.1%} ({g['k']}/{g['n']}) — ${g['avg_cost']:.5f}/ep")
            improvement = g['rate'] - b['rate']
            print(f"    Δ correctness: {improvement:+.1%}")

    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
