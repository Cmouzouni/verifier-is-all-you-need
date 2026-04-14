"""T2.2 — ARC capability floor measurement.

The question: can cheap models even solve ARC-AGI problems with unlimited tries?
If not, the synthesis-completed game has nothing to recognize and ARC is off the
table for cheap-collective architectures.

Design:
  - 30 ARC-AGI training problems (sampled with seed 42)
  - 3 Qwen tiers: qwen-7b, qwen-22b, qwen-17b
  - For each (model, problem), draw N=16 independent samples
  - Compute best-of-N solve rate at N ∈ {1, 4, 8, 16}
  - Verify by exact grid match

Decision rule:
  - qwen-7b best-of-16 ≥ 20%: ARC in scope, full game next
  - qwen-22b best-of-16 ≥ 20% but qwen-7b < 20%: needs tiered escalation
  - All tiers near 0%: ARC is a capability problem, pivot to MATH/MuSR
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, parse_grid_answer


# ── Configuration ──────────────────────────────────────────────────────
N_PROBLEMS = 30
N_SAMPLES = 16  # samples per (model, problem)
TEMPERATURE = 0.8  # higher temp for sample diversity
MAX_TOKENS = 2048  # ARC outputs can be larger grids
MODELS = ["qwen-7b", "qwen-22b", "qwen-17b"]
N_VALUES = [1, 4, 8, 16]


def run_one_sample(client: LLMClient, task) -> dict:
    """Single attempt at one ARC task."""
    sys_prompt = (
        "You are a visual-spatial reasoning expert. You will be shown several "
        "input-output grid pairs that follow the same transformation rule. "
        "Your job is to figure out the rule and apply it to a new input grid. "
        "Output the resulting grid in the exact same format as the examples."
    )
    usr_prompt = task.format_prompt()
    try:
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        correct = task.check(resp.content)
        parsed = parse_grid_answer(resp.content) is not None
        return {
            "correct": correct,
            "parsed": parsed,
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        }
    except Exception as e:
        return {"correct": False, "parsed": False, "tokens": 0, "cost": 0, "error": str(e)[:80]}


def run_model_on_tasks(client: LLMClient, tasks: list, n_samples: int) -> dict:
    """Run N samples on each task in parallel; returns per-task sample lists."""
    results = {t.id: [] for t in tasks}

    jobs = []
    for task in tasks:
        for _ in range(n_samples):
            jobs.append(task)

    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_task = {pool.submit(run_one_sample, client, t): t for t in jobs}
        completed = 0
        for f in as_completed(future_to_task):
            t = future_to_task[f]
            try:
                results[t.id].append(f.result())
            except Exception as e:
                results[t.id].append({"correct": False, "parsed": False, "tokens": 0, "cost": 0, "error": str(e)[:80]})
            completed += 1
            if completed % 60 == 0:
                total_correct = sum(1 for tid, samples in results.items() for s in samples if s["correct"])
                total_parsed = sum(1 for tid, samples in results.items() for s in samples if s.get("parsed"))
                cost = client.tracker.total_cost
                print(f"  [{completed}/{len(jobs)}] {total_correct} correct, {total_parsed} parsed, cost ${cost:.4f}", flush=True)

    return results


def best_of_n(samples: list[dict], n: int) -> bool:
    """Did at least one of the first N samples succeed?"""
    return any(s["correct"] for s in samples[:n])


def main(dry_run: bool = False):
    print("=" * 70)
    print("T2.2 — ARC CAPABILITY FLOOR MEASUREMENT")
    print("=" * 70)

    print(f"\nLoading ARC-AGI...")
    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  {len(tasks)} tasks loaded")
    print(f"\nModels: {MODELS}")
    print(f"N samples per (model, task): {N_SAMPLES}")
    print(f"Best-of-N values: {N_VALUES}")
    n_calls_total = len(MODELS) * len(tasks) * N_SAMPLES
    print(f"Total API calls: {n_calls_total}")

    if dry_run:
        # ARC inputs are large; estimate ~2000 tokens in, ~1500 tokens out
        # Average across model prices
        avg_in_cost = 0.40 * 2000 / 1e6   # weighted avg of model prices
        avg_out_cost = 1.40 * 1500 / 1e6
        est = n_calls_total * (avg_in_cost + avg_out_cost)
        print(f"\n[DRY RUN] Estimated cost: ${est:.2f}")
        print(f"[DRY RUN] (rough — large grids → many tokens)")
        return

    output_dir = RESULTS_DIR / "t22_arc_floor"
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    t_start = time.time()

    for model in MODELS:
        print(f"\n--- Model: {model} ---", flush=True)
        client = LLMClient(model)

        results = run_model_on_tasks(client, tasks, N_SAMPLES)

        # Compute best-of-N rates
        rates = {}
        for n in N_VALUES:
            solved = sum(1 for tid, samples in results.items() if best_of_n(samples, n))
            rates[n] = solved / len(tasks)

        # Diversity diagnostic: how many distinct answers per task at N=16
        distinct_counts = []
        for tid, samples in results.items():
            distinct_correct = sum(1 for s in samples if s["correct"])
            distinct_counts.append(distinct_correct)

        all_results[model] = {
            "rates": rates,
            "n_correct_per_task": distinct_counts,
            "total_cost": client.tracker.total_cost,
            "total_tokens": client.tracker.total_tokens,
            "raw": {tid: samples for tid, samples in results.items()},
        }

        print(f"  best-of-1:  {rates[1]:.1%}")
        print(f"  best-of-4:  {rates[4]:.1%}")
        print(f"  best-of-8:  {rates[8]:.1%}")
        print(f"  best-of-16: {rates[16]:.1%}")
        print(f"  cost: ${client.tracker.total_cost:.4f}")

    elapsed = time.time() - t_start

    # ── Decision rule ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"T2.2 RESULTS — ARC Capability Floor")
    print(f"{'='*70}")

    print(f"\n  Model       N=1     N=4     N=8     N=16")
    for model in MODELS:
        rates = all_results[model]["rates"]
        print(f"  {model:11s} {rates[1]:6.1%}  {rates[4]:6.1%}  {rates[8]:6.1%}  {rates[16]:6.1%}")

    print(f"\n--- DECISION ---")
    qwen7b_best = all_results["qwen-7b"]["rates"][16]
    qwen22b_best = all_results["qwen-22b"]["rates"][16]
    qwen17b_best = all_results["qwen-17b"]["rates"][16]

    if qwen7b_best >= 0.20:
        print(f"  ✓ qwen-7b best-of-16 = {qwen7b_best:.1%} ≥ 20%")
        print(f"  → ARC is in scope. Run the cheap-collective game on ARC next.")
    elif qwen22b_best >= 0.20:
        print(f"  ~ qwen-7b best-of-16 = {qwen7b_best:.1%} < 20% but qwen-22b = {qwen22b_best:.1%} ≥ 20%")
        print(f"  → ARC needs tiered escalation. Cheap tier alone is insufficient.")
    elif qwen17b_best >= 0.20:
        print(f"  ~ Only qwen-17b achieves ≥20%: {qwen17b_best:.1%}")
        print(f"  → ARC requires the frontier tier. The cheap-collective story does not apply here.")
    else:
        print(f"  ✗ All Qwen tiers below 20% best-of-16 (max: {max(qwen7b_best, qwen22b_best, qwen17b_best):.1%})")
        print(f"  → ARC is a capability problem, not a coverage problem.")
        print(f"  → Pivot to MATH or MuSR for the harder-benchmark validation.")

    total_cost = sum(all_results[m]["total_cost"] for m in MODELS)
    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save (excluding raw to keep file size small)
    save_data = {
        "experiment": "t22_arc_floor",
        "n_problems": len(tasks),
        "n_samples": N_SAMPLES,
        "results": {
            m: {k: v for k, v in d.items() if k != "raw"}
            for m, d in all_results.items()
        },
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }
    json.dump(save_data, open(output_dir / "results.json", "w"), indent=2, default=str)

    # Save full raw separately
    raw_data = {m: d["raw"] for m, d in all_results.items()}
    json.dump(raw_data, open(output_dir / "raw.json", "w"), indent=2, default=str)

    print(f"\n  Results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
