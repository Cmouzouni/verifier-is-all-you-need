"""Tier 1.1 — Hard benchmark replication on GSM8K.

Tests whether the synthesis-completed cost-efficiency win generalizes from our
Phase A multi-approach tasks to a standard public benchmark with external
verification.

Design:
  - 50 GSM8K problems (medium difficulty, sampled with seed=42)
  - 5 game configurations (top performers from main sweep)
  - 3 single-model baselines (qwen-7b, qwen-22b, qwen-17b)
  - 2 episodes per problem per condition (cost control)

Top 5 game configs from main sweep (cheapest 100% from Table 2):
  1. all-7b × insight-synth × γ=1
  2. propose7b-synth22b × full-debate × γ=0.5
  3. all-22b × full-debate × γ=0.5
  4. propose22b-crit7b × insight-synth × γ=1
  5. all-7b × full-debate × γ=0.5
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.runner import _extract_answer
from src.config import RESULTS_DIR
from experiments.exp_full_sweep_parallel import run_episode_parallel
from experiments.exp_full_sweep import MODEL_CONFIGS, TOPOLOGIES
from tasks.gsm8k_tasks import load_gsm8k


# ── Experiment configuration ──────────────────────────────────────────
N_PROBLEMS = 50
EPISODES_PER_PROBLEM = 2
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Top 5 game configurations from the main sweep (Table 2)
TIER1_CONFIGS = [
    ("all-7b", "all-insight-synth", 1.0),
    ("propose7b-synth22b", "full-debate", 0.5),
    ("all-22b", "full-debate", 0.5),
    ("propose22b-crit7b", "all-insight-synth", 1.0),
    ("all-7b", "full-debate", 0.5),
]


def run_baseline(client: LLMClient, task) -> dict:
    """Single-agent baseline: one model, one shot per problem."""
    sys_prompt = "You are a math word problem solver. Solve step by step. State your final answer as a number on a line starting with 'ANSWER: '."
    usr_prompt = f"Problem: {task.problem}"
    try:
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        answer = _extract_answer(resp.content)
        correct = task.check(answer)
        return {
            "task_id": task.id,
            "answer": answer,
            "correct": correct,
            "cost": resp.cost_usd,
            "tokens": resp.input_tokens + resp.output_tokens,
        }
    except Exception as e:
        return {"task_id": task.id, "answer": "", "correct": False, "cost": 0, "tokens": 0, "error": str(e)[:80]}


def run_baselines_parallel(clients: dict[str, LLMClient], tasks: list, model_keys: list[str]) -> dict:
    """Run all baselines in parallel."""
    results = defaultdict(list)
    for model_key in model_keys:
        client = clients[model_key]
        print(f"  Baseline: {model_key} ({len(tasks)} tasks)...", flush=True)
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(run_baseline, client, t) for t in tasks]
            for f in as_completed(futures):
                results[model_key].append(f.result())
        n = len(results[model_key])
        k = sum(r["correct"] for r in results[model_key])
        cost = sum(r["cost"] for r in results[model_key])
        print(f"    {model_key}: {k}/{n} ({k/n:.1%}) — ${cost:.4f}", flush=True)
    return dict(results)


def run_game_condition(
    clients: dict, tasks: list, mc_name: str, tp_name: str, gamma: float,
    episodes_per_task: int = 2,
) -> list:
    """Run one game configuration on all tasks in parallel."""
    mc = MODEL_CONFIGS[mc_name]
    tp = TOPOLOGIES[tp_name]
    rng = random.Random(42)

    jobs = []
    for task in tasks:
        for ep in range(episodes_per_task):
            jobs.append((task, rng.randint(0, 2**31)))

    records = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(
                run_episode_parallel,
                task=task, gamma=gamma,
                model_config_name=mc_name, model_config=mc,
                topology_name=tp_name, topology=tp,
                clients=clients, seed=seed,
            ): (task, seed)
            for task, seed in jobs
        }
        for f in as_completed(futures):
            try:
                records.append(f.result())
            except Exception as e:
                print(f"    [ERROR] {e}", flush=True)
    return records


def main(dry_run: bool = False):
    print("=" * 70)
    print("TIER 1.1 — HARD BENCHMARK REPLICATION (GSM8K)")
    print("=" * 70)

    tasks = load_gsm8k(n=N_PROBLEMS, seed=42)
    print(f"Loaded {len(tasks)} GSM8K tasks")
    print(f"Game configs: {len(TIER1_CONFIGS)}")
    print(f"Episodes per (config, task): {EPISODES_PER_PROBLEM}")
    total_game_episodes = len(TIER1_CONFIGS) * len(tasks) * EPISODES_PER_PROBLEM
    total_baseline_episodes = 3 * len(tasks)
    print(f"Total: {total_game_episodes} game episodes + {total_baseline_episodes} baselines")

    if dry_run:
        # Estimate: avg 4 calls per game episode at ~600 tokens each
        avg_call_cost = 0.4 * 800 / 1e6  # mix of 7b/22b
        game_cost = total_game_episodes * 4 * avg_call_cost
        baseline_cost = total_baseline_episodes * avg_call_cost
        print(f"\n[DRY RUN] Estimated: ${game_cost + baseline_cost:.2f}")
        print(f"  Game episodes: ${game_cost:.2f}")
        print(f"  Baselines: ${baseline_cost:.2f}")
        return

    # ── Initialize clients ─────────────────────────────────────────────
    needed = set()
    for mc_name, _, _ in TIER1_CONFIGS:
        for k, v in MODEL_CONFIGS[mc_name].items():
            if k != "description":
                needed.add(v)
    for m in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        needed.add(m)

    clients = {m: LLMClient(model_key=m) for m in needed}
    print(f"Clients: {list(clients.keys())}")

    t_start = time.time()
    output_dir = RESULTS_DIR / "tier1_gsm8k"
    output_dir.mkdir(exist_ok=True)

    # ── Phase 1: Baselines ─────────────────────────────────────────────
    print("\n--- BASELINES ---")
    baselines = run_baselines_parallel(clients, tasks, ["qwen-7b", "qwen-22b", "qwen-17b"])

    # Save baselines immediately
    json.dump(
        {"baselines": {m: rs for m, rs in baselines.items()}},
        open(output_dir / "baselines.json", "w"),
        indent=2,
        default=str,
    )

    # ── Phase 2: Game configurations ───────────────────────────────────
    print("\n--- GAME CONFIGS ---")
    all_game_records = []
    for i, (mc, tp, g) in enumerate(TIER1_CONFIGS, 1):
        print(f"\n  [{i}/{len(TIER1_CONFIGS)}] {mc} × {tp} γ={g}", flush=True)
        records = run_game_condition(clients, tasks, mc, tp, g, EPISODES_PER_PROBLEM)
        if records:
            k = sum(r.correct for r in records)
            n = len(records)
            cost = sum(r.total_cost_usd for r in records) / n
            print(f"    -> {k}/{n} ({k/n:.1%}) — avg cost/ep ${cost:.5f}", flush=True)
            all_game_records.extend([{**asdict(r), "config_label": f"{mc}|{tp}|γ={g}"} for r in records])

    # ── Save complete results ──────────────────────────────────────────
    elapsed = time.time() - t_start
    total_cost = sum(c.tracker.total_cost for c in clients.values())

    data = {
        "experiment": "tier1_gsm8k",
        "n_problems": len(tasks),
        "configs": [{"model_config": mc, "topology": tp, "gamma": g} for mc, tp, g in TIER1_CONFIGS],
        "summary": {
            "total_episodes": len(all_game_records),
            "total_cost_usd": total_cost,
            "elapsed_s": elapsed,
        },
        "baselines": {m: rs for m, rs in baselines.items()},
        "game_episodes": all_game_records,
    }
    (output_dir / "results.json").write_text(json.dumps(data, indent=2, default=str))

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TIER 1.1 RESULTS — GSM8K")
    print(f"{'='*70}")

    print(f"\n--- BASELINES ---")
    for m in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        rs = baselines[m]
        k = sum(r["correct"] for r in rs)
        n = len(rs)
        cost = sum(r["cost"] for r in rs)
        print(f"  {m:12s}: {k/n:.1%} ({k}/{n}) — total ${cost:.4f}")

    print(f"\n--- GAME CONFIGS ---")
    by_label = defaultdict(list)
    for r in all_game_records:
        by_label[r["config_label"]].append(r)
    for label, recs in by_label.items():
        k = sum(r["correct"] for r in recs)
        n = len(recs)
        cost = sum(r["total_cost_usd"] for r in recs) / n
        print(f"  {label:50s}: {k/n:.1%} ({k}/{n}) cost=${cost:.5f}")

    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
