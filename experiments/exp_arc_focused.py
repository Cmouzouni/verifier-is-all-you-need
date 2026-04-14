"""ARC Focused — minimal high-priority experiments while qwen-22b is flaky.

Priorities:
  1. qwen-17b WITH thinking baseline (the fair frontier number — currently MISSING)
  2. Best-of-N qwen-17b with thinking (maximum capability + parallelism)
  3. DeepSeek-V3.1 single baseline (alternative frontier)
  4. Heterogeneous: 4× qwen-17b proposers + qwen-17b synth (frontier game)
  5. Heterogeneous: 4× DeepSeek + qwen-17b synth (mixed frontier game)

The avoid-qwen-22b strategy: use qwen-17b and DeepSeek which are working reliably.
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
from tasks.arc_tasks import load_arc_tasks, parse_grid_answer, ARCTask
from experiments.exp_arc_crack import (
    call_arc, synthesize_over_candidates, FRAMINGS, parse_canonical, check_canonical,
)


N_PROBLEMS = 30
WORKERS = 8


def run_baseline_with_thinking(name: str, client: LLMClient, tasks: list[ARCTask]) -> dict:
    """Single-shot baseline. Frontier models use thinking via call_arc auto-detect."""
    print(f"\n--- BASELINE: {name} ---", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(call_arc, client, t) for t in tasks]
        for f in as_completed(futures):
            records.append(f.result())
    k = sum(r["correct"] for r in records)
    n = len(records)
    cost = sum(r["cost"] for r in records)
    print(f"  → {name}: {k}/{n} ({k/n:.1%}) cost ${cost:.4f}", flush=True)
    return {"name": name, "k": k, "n": n, "rate": k / n, "cost": cost, "type": "baseline"}


def run_best_of_n(name: str, client: LLMClient, tasks: list[ARCTask], n_samples: int) -> dict:
    """Best-of-N (oracle) — count problems where ANY of N samples is correct."""
    print(f"\n--- BEST-OF-{n_samples}: {name} ---", flush=True)
    results_by_task = defaultdict(list)
    jobs = []
    for t in tasks:
        for i in range(n_samples):
            jobs.append((t, i))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_job = {pool.submit(call_arc, client, t, framing=FRAMINGS[i % len(FRAMINGS)]): t.id
                          for t, i in jobs}
        completed = 0
        for f in as_completed(future_to_job):
            tid = future_to_job[f]
            r = f.result()
            results_by_task[tid].append(r)
            completed += 1
            if completed % 30 == 0:
                tasks_with_any = sum(1 for tid, samples in results_by_task.items() if any(s["correct"] for s in samples))
                print(f"  [{completed}/{len(jobs)}] {tasks_with_any} tasks with ≥1 correct so far", flush=True)

    per_task_correct = []
    total_cost = 0
    for t in tasks:
        recs = results_by_task[t.id]
        per_task_correct.append(any(r["correct"] for r in recs))
        total_cost += sum(r["cost"] for r in recs)

    k = sum(per_task_correct)
    print(f"  → {name} best-of-{n_samples}: {k}/{len(tasks)} ({k/len(tasks):.1%}) cost ${total_cost:.4f}", flush=True)
    return {
        "name": f"{name}_best_of_{n_samples}",
        "k": k, "n": len(tasks), "rate": k / len(tasks),
        "cost": total_cost, "type": "oracle",
    }


def run_synth_game(name: str, prop_clients: list[LLMClient], synth_client: LLMClient,
                    tasks: list[ARCTask]) -> dict:
    """N proposers (one per prop_clients) + 1 synthesizer."""
    n = len(prop_clients)
    print(f"\n--- SYNTH GAME N={n}: {name} ---", flush=True)
    synth_results = []

    def run_one_episode(task):
        with ThreadPoolExecutor(max_workers=n) as inner:
            futures = [
                inner.submit(call_arc, prop_clients[i], task, framing=FRAMINGS[i % len(FRAMINGS)])
                for i in range(n)
            ]
            proposals = [f.result() for f in futures]
        synth = synthesize_over_candidates(synth_client, task, proposals)
        return {
            "task_id": task.id,
            "n_proposers_correct": sum(1 for p in proposals if p["correct"]),
            "synth_correct": synth["correct"],
            "total_cost": sum(p["cost"] for p in proposals) + synth["cost"],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one_episode, t) for t in tasks]
        completed = 0
        for f in as_completed(futures):
            try:
                synth_results.append(f.result())
                completed += 1
                if completed % 5 == 0:
                    k = sum(r["synth_correct"] for r in synth_results)
                    any_k = sum(1 for r in synth_results if r["n_proposers_correct"] > 0)
                    cost = sum(r["total_cost"] for r in synth_results)
                    print(f"  [{completed}/{len(tasks)}] synth={k}/{completed} any-prop={any_k}/{completed} cost ${cost:.4f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)

    k = sum(r["synth_correct"] for r in synth_results)
    n_eps = len(synth_results)
    any_k = sum(1 for r in synth_results if r["n_proposers_correct"] > 0)
    cost = sum(r["total_cost"] for r in synth_results)
    print(f"  → {name}: synth={k}/{n_eps} ({k/n_eps:.1%}), any-prop={any_k}/{n_eps} ({any_k/n_eps:.1%}), cost ${cost:.4f}", flush=True)
    return {
        "name": name,
        "k": k, "n": n_eps, "rate": k / n_eps,
        "any_proposer_rate": any_k / n_eps,
        "cost": cost,
        "type": "synth",
    }


def main():
    print("=" * 70)
    print("ARC FOCUSED — high-priority experiments avoiding flaky qwen-22b")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")

    print(f"\nLoading ARC-AGI...")
    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  {len(tasks)} tasks loaded")

    clients = {
        "qwen-7b": LLMClient("qwen-7b"),
        "qwen-17b": LLMClient("qwen-17b"),
        "llama-70b": LLMClient("llama-70b"),
        "deepseek": LLMClient("deepseek"),
    }

    output_dir = RESULTS_DIR / "arc_focused"
    output_dir.mkdir(exist_ok=True)
    t_start = time.time()
    all_results = {}

    def save():
        json.dump({
            "experiment": "arc_focused",
            "n_problems": len(tasks),
            "results": all_results,
            "elapsed_s": time.time() - t_start,
        }, open(output_dir / "results.json", "w"), indent=2, default=str)

    # ── Phase 1: Critical baselines ────────────────────────────────────
    # Most important: qwen-17b WITH thinking (the fair frontier number)
    r = run_baseline_with_thinking("qwen-17b (with thinking)", clients["qwen-17b"], tasks)
    all_results[r["name"]] = r
    save()

    # DeepSeek baseline (alternative frontier)
    r = run_baseline_with_thinking("deepseek-v3.1", clients["deepseek"], tasks)
    all_results[r["name"]] = r
    save()

    # Llama-70b baseline (large fast alternative)
    r = run_baseline_with_thinking("llama-70b", clients["llama-70b"], tasks)
    all_results[r["name"]] = r
    save()

    # ── Phase 2: Best-of-N with the strongest model ────────────────────
    # qwen-17b best-of-8 (massive parallelism + frontier)
    r = run_best_of_n("qwen-17b (with thinking)", clients["qwen-17b"], tasks, n_samples=8)
    all_results[r["name"]] = r
    save()

    # deepseek best-of-8
    r = run_best_of_n("deepseek-v3.1", clients["deepseek"], tasks, n_samples=8)
    all_results[r["name"]] = r
    save()

    # ── Phase 3: Heterogeneous game with frontier proposers ────────────
    # 4× qwen-17b proposers + qwen-17b synth (pure frontier game)
    r = run_synth_game(
        "FRONTIER-GAME__4x17b_synth17b",
        [clients["qwen-17b"]] * 4,
        clients["qwen-17b"],
        tasks,
    )
    all_results[r["name"]] = r
    save()

    # 2× qwen-17b + 2× deepseek proposers + qwen-17b synth (mixed frontier game)
    r = run_synth_game(
        "MIXED-FRONTIER__2x17b_2xds_synth17b",
        [clients["qwen-17b"]] * 2 + [clients["deepseek"]] * 2,
        clients["qwen-17b"],
        tasks,
    )
    all_results[r["name"]] = r
    save()

    # ── Final summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_cost = sum(c.tracker.total_cost for c in clients.values())

    print(f"\n{'='*70}")
    print(f"ARC FOCUSED RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Name':50s} {'Type':10s} {'Rate':>8s} {'Cost':>10s}")
    print(f"  {'-'*50} {'-'*10} {'-'*8} {'-'*10}")
    sorted_results = sorted(all_results.values(), key=lambda x: -x["rate"])
    for r in sorted_results:
        marker = "★" if r["rate"] >= 0.30 else " "
        print(f"  {marker} {r['name']:48s} {r['type']:10s} {r['rate']:>7.1%} ${r['cost']:>8.4f}")

    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    save()


if __name__ == "__main__":
    main()
