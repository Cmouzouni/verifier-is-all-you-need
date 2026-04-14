"""Clean re-measurement of single-model ARC baselines.

The audit found that the paper's "qwen-22b 30% baseline" was fabricated — the
actual prior measurement was 3/30 = 10%, and the 30% number came from a
hardcoded reference in a script.

This experiment cleanly re-measures all three Qwen tier baselines on the same
30 ARC-AGI tasks (seed=42). It uses:
  - 5 attempts per task per model (n_samples) — accounts for sampling noise
  - Uses thinking on qwen-17b (the fair frontier baseline)
  - Same task set as the rest of the campaign

Reports per-attempt rate, best-of-5 (oracle), and majority vote.

Cost estimate: ~$0.50 (3 models × 30 tasks × 5 samples = 450 calls)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, parse_grid_answer
from experiments.exp_arc_crack import call_arc, parse_canonical, check_canonical


N_PROBLEMS = 30
N_SAMPLES = 5
WORKERS = 4
SEED = 42


def measure_model(model_key: str, tasks, client: LLMClient) -> dict:
    """Run N_SAMPLES attempts per task, report per-attempt + best-of-N + majority."""
    print(f"\n--- {model_key} (n={N_SAMPLES} per task) ---", flush=True)
    by_task = defaultdict(list)

    jobs = [(t, i) for t in tasks for i in range(N_SAMPLES)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(call_arc, client, t): t.id for t, _ in jobs}
        completed = 0
        for f in as_completed(futures):
            tid = futures[f]
            try:
                r = f.result()
                by_task[tid].append(r)
            except Exception as e:
                by_task[tid].append({"correct": False, "cost": 0, "content": "", "error": str(e)[:80]})
            completed += 1
            if completed % 30 == 0:
                tasks_with_correct = sum(1 for tid, rs in by_task.items() if any(r.get("correct") for r in rs))
                cost = sum(sum(r.get("cost", 0) for r in rs) for rs in by_task.values())
                print(f"  [{completed}/{len(jobs)}] {tasks_with_correct} tasks have ≥1 correct, cost ${cost:.4f}", flush=True)

    # Compute metrics:
    # 1. Per-attempt rate (single-shot baseline)
    # 2. Best-of-N rate (any sample correct)
    # 3. Majority vote rate (most-common canonical grid is correct)
    n_attempts = 0
    n_attempts_correct = 0
    n_tasks_any_correct = 0
    n_tasks_majority_correct = 0
    total_cost = 0

    for t in tasks:
        rs = by_task[t.id]
        for r in rs:
            n_attempts += 1
            if r.get("correct"):
                n_attempts_correct += 1
            total_cost += r.get("cost", 0)
        if any(r.get("correct") for r in rs):
            n_tasks_any_correct += 1
        # Majority vote on canonical grids
        cans = [parse_canonical(r.get("content", "")) for r in rs]
        votes = Counter(c for c in cans if c)
        if votes:
            top, _ = votes.most_common(1)[0]
            if check_canonical(top, t):
                n_tasks_majority_correct += 1

    per_attempt = n_attempts_correct / n_attempts if n_attempts else 0
    any_correct_rate = n_tasks_any_correct / len(tasks)
    majority_rate = n_tasks_majority_correct / len(tasks)

    print(f"  → per-attempt: {n_attempts_correct}/{n_attempts} = {per_attempt:.1%}", flush=True)
    print(f"  → any-correct (best-of-{N_SAMPLES}): {n_tasks_any_correct}/{len(tasks)} = {any_correct_rate:.1%}", flush=True)
    print(f"  → majority vote: {n_tasks_majority_correct}/{len(tasks)} = {majority_rate:.1%}", flush=True)
    print(f"  → cost: ${total_cost:.4f}", flush=True)

    return {
        "model": model_key,
        "n_tasks": len(tasks),
        "n_samples_per_task": N_SAMPLES,
        "per_attempt": {
            "k": n_attempts_correct,
            "n": n_attempts,
            "rate": per_attempt,
        },
        "any_correct": {
            "k": n_tasks_any_correct,
            "n": len(tasks),
            "rate": any_correct_rate,
        },
        "majority_vote": {
            "k": n_tasks_majority_correct,
            "n": len(tasks),
            "rate": majority_rate,
        },
        "total_cost": total_cost,
        "raw": {tid: rs for tid, rs in by_task.items()},
    }


def main():
    print("=" * 70)
    print("ARC BASELINE RE-MEASUREMENT")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")
    print(f"  Samples per task: {N_SAMPLES}")
    print(f"  Total calls per model: {N_PROBLEMS * N_SAMPLES}")
    print(f"  Models: qwen-7b, qwen-22b, qwen-17b")

    tasks = load_arc_tasks(n=N_PROBLEMS, seed=SEED)
    print(f"\nLoaded {len(tasks)} tasks (seed={SEED})")

    output_dir = RESULTS_DIR / "arc_baseline_remeasure"
    output_dir.mkdir(exist_ok=True)

    t_start = time.time()
    results = {}

    for model_key in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        client = LLMClient(model_key)
        result = measure_model(model_key, tasks, client)
        results[model_key] = result
        # Save incrementally
        save = {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in results.items()}
        json.dump({"results": save}, open(output_dir / "results.json", "w"), indent=2, default=str)
        # Save raw separately
        json.dump({k: v.get("raw", {}) for k, v in results.items()},
                  open(output_dir / "raw.json", "w"), indent=2, default=str)

    elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"FINAL RE-MEASUREMENT")
    print(f"{'='*70}")
    print(f"\n  {'Model':12s} {'per-attempt':>14s} {'best-of-5':>12s} {'majority':>12s} {'cost':>10s}")
    print(f"  {'-'*12} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")
    for m, r in results.items():
        pa = r['per_attempt']
        ac = r['any_correct']
        mv = r['majority_vote']
        cost = r['total_cost']
        print(f"  {m:12s} {pa['rate']:7.1%} ({pa['k']:3d}/{pa['n']:3d})  {ac['rate']:7.1%}      {mv['rate']:7.1%}      ${cost:.4f}")

    print(f"\n  Total elapsed: {elapsed:.0f}s")
    print(f"\n  CRITICAL: paper currently claims qwen-22b ARC baseline = 30%")
    qwen22_pa = results['qwen-22b']['per_attempt']['rate']
    print(f"  ACTUAL re-measurement (per-attempt): {qwen22_pa:.1%}")
    print(f"  Difference from paper claim: {(qwen22_pa - 0.30)*100:+.1f}pp")


if __name__ == "__main__":
    main()
