"""ARC Crack v2 — using qwen-22b as the proposer (NOT qwen-7b).

The key insight from EXP-A1: cheap-7b proposals are below the capability floor on ARC.
Even 16 attempts can't generate correct candidates on most ARC problems.

But we just discovered: **qwen-22b single-shot achieves 30% on ARC-30.**

So the right strategy is to use qwen-22b for proposals (where capability matters)
and either qwen-22b or qwen-17b for synthesis (where recognition matters).

Architectures tested:
  EXP-V2-A: 8× qwen-22b proposers + qwen-17b synth (with thinking)
  EXP-V2-B: 16× qwen-22b proposers (best-of-16 with consensus)
  EXP-V2-C: 8× qwen-22b proposers + qwen-22b synth (no frontier needed?)
  EXP-V2-D: oracle best-of-16 of qwen-22b (theoretical ceiling)
  EXP-V2-E: oracle best-of-32 of qwen-22b (more samples, theoretical ceiling)

Plus baselines (using thinking when applicable):
  - qwen-22b single (9/30 = 30%, already measured)
  - qwen-17b single with thinking (pending)

Goal: ≥40% correctness on ARC-30, beating the qwen-22b 30% baseline meaningfully.
"""

from __future__ import annotations

import json
import random
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
    call_arc, synthesize_over_candidates, FRAMINGS, ARC_SYSTEM_PROMPT,
    parse_canonical, check_canonical,
)


# ── Configuration ──────────────────────────────────────────────────────
N_PROBLEMS = 30
WORKERS = 30


def exp_synth_22b_proposers(name: str, n_proposers: int, synth_model: str,
                              tasks: list[ARCTask], clients: dict) -> dict:
    """N qwen-22b proposers + 1 synthesizer."""
    print(f"\n--- {name}: {n_proposers}× qwen-22b proposers + {synth_model} synth ---", flush=True)
    synth_results = []

    def run_one(task):
        # Run N qwen-22b proposers in parallel, each with a different framing
        with ThreadPoolExecutor(max_workers=n_proposers) as inner:
            futures = [
                inner.submit(call_arc, clients["qwen-22b"], task, framing=FRAMINGS[i % len(FRAMINGS)])
                for i in range(n_proposers)
            ]
            proposals = [f.result() for f in futures]

        # Synthesize
        synth = synthesize_over_candidates(clients[synth_model], task, proposals)

        prop_cost = sum(p["cost"] for p in proposals)
        return {
            "task_id": task.id,
            "n_proposers_correct": sum(1 for p in proposals if p["correct"]),
            "synth_correct": synth["correct"],
            "prop_cost": prop_cost,
            "synth_cost": synth["cost"],
            "total_cost": prop_cost + synth["cost"],
        }

    # Run multiple episodes in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
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
    n = len(synth_results)
    any_k = sum(1 for r in synth_results if r["n_proposers_correct"] > 0)
    cost = sum(r["total_cost"] for r in synth_results)
    print(f"  → {name}: synth={k}/{n} ({k/n:.1%}), any-prop={any_k}/{n} ({any_k/n:.1%}), cost ${cost:.4f}", flush=True)
    return {
        "name": name,
        "k": k, "n": n, "rate": k / n,
        "any_proposer_rate": any_k / n,
        "cost": cost,
        "type": "synth-22b",
    }


def exp_consensus_22b(name: str, n_samples: int, tasks: list[ARCTask], client: LLMClient) -> dict:
    """N samples from qwen-22b, take the most common answer (no synthesizer)."""
    print(f"\n--- {name}: best-of-{n_samples} qwen-22b consensus ---", flush=True)

    def run_one(task):
        # N samples in parallel
        with ThreadPoolExecutor(max_workers=n_samples) as inner:
            futures = [
                inner.submit(call_arc, client, task, framing=FRAMINGS[i % len(FRAMINGS)])
                for i in range(n_samples)
            ]
            samples = [f.result() for f in futures]
        # Consensus by canonical grid
        canonicals = [parse_canonical(s["content"]) for s in samples]
        counts = Counter(c for c in canonicals if c)
        if counts:
            top, _ = counts.most_common(1)[0]
            consensus_correct = check_canonical(top, task)
        else:
            consensus_correct = False
        any_correct = any(s["correct"] for s in samples)
        return {
            "task_id": task.id,
            "consensus_correct": consensus_correct,
            "any_correct": any_correct,
            "n_samples_correct": sum(1 for s in samples if s["correct"]),
            "cost": sum(s["cost"] for s in samples),
        }

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        completed = 0
        for f in as_completed(futures):
            try:
                records.append(f.result())
                completed += 1
                if completed % 5 == 0:
                    consensus_k = sum(r["consensus_correct"] for r in records)
                    any_k = sum(r["any_correct"] for r in records)
                    cost = sum(r["cost"] for r in records)
                    print(f"  [{completed}/{len(tasks)}] consensus={consensus_k}/{completed} any={any_k}/{completed} cost ${cost:.4f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)

    consensus_k = sum(r["consensus_correct"] for r in records)
    any_k = sum(r["any_correct"] for r in records)
    n = len(records)
    cost = sum(r["cost"] for r in records)
    print(f"  → {name}: consensus={consensus_k}/{n} ({consensus_k/n:.1%}), "
          f"any={any_k}/{n} ({any_k/n:.1%}), cost ${cost:.4f}", flush=True)
    return {
        "name": name,
        "k": consensus_k, "n": n, "rate": consensus_k / n,
        "any_correct_rate": any_k / n,
        "cost": cost,
        "type": "consensus-22b",
    }


def main(dry_run: bool = False, only: list[str] | None = None):
    print("=" * 70)
    print("ARC CRACK v2 — using qwen-22b as proposer (target: ≥40%)")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")
    print(f"  Workers: {WORKERS}")

    if dry_run:
        # Rough cost: each architecture is N × qwen-22b + maybe synth × qwen-17b
        n = N_PROBLEMS
        # EXP-V2-A: 8 prop + 1 synth × 30 tasks ~$0.0028 per call avg
        # 8 × 0.001 + 0.005 = 0.013/ep × 30 = 0.40
        configs_cost = {
            "synth-A_8x22b_synth17b": n * (8 * 0.0015 + 0.008),  # ~$0.6
            "synth-C_8x22b_synth22b": n * (9 * 0.0015),  # ~$0.4
            "consensus-D_16x22b": n * 16 * 0.0015,  # ~$0.7
            "consensus-E_32x22b": n * 32 * 0.0015,  # ~$1.4
        }
        total = sum(configs_cost.values())
        print(f"\n[DRY RUN] estimated total: ${total:.2f}")
        for k, v in configs_cost.items():
            print(f"  {k}: ${v:.2f}")
        return

    print(f"\nLoading ARC-AGI...")
    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  {len(tasks)} tasks loaded")

    clients = {
        "qwen-22b": LLMClient("qwen-22b"),
        "qwen-17b": LLMClient("qwen-17b"),
    }

    output_dir = RESULTS_DIR / "arc_crack_v2"
    output_dir.mkdir(exist_ok=True)
    t_start = time.time()
    all_results = {}

    def should_run(exp: str) -> bool:
        return only is None or exp in only

    if should_run("synth_a"):
        r = exp_synth_22b_proposers("EXP-V2-A__8x22b_synth17b", 8, "qwen-17b", tasks, clients)
        all_results[r["name"]] = r
        json.dump({"results": all_results}, open(output_dir / "partial.json", "w"), indent=2, default=str)

    if should_run("synth_c"):
        r = exp_synth_22b_proposers("EXP-V2-C__8x22b_synth22b", 8, "qwen-22b", tasks, clients)
        all_results[r["name"]] = r
        json.dump({"results": all_results}, open(output_dir / "partial.json", "w"), indent=2, default=str)

    if should_run("consensus_d"):
        r = exp_consensus_22b("EXP-V2-D__16x22b_consensus", 16, tasks, clients["qwen-22b"])
        all_results[r["name"]] = r
        json.dump({"results": all_results}, open(output_dir / "partial.json", "w"), indent=2, default=str)

    if should_run("consensus_e"):
        r = exp_consensus_22b("EXP-V2-E__32x22b_consensus", 32, tasks, clients["qwen-22b"])
        all_results[r["name"]] = r
        json.dump({"results": all_results}, open(output_dir / "partial.json", "w"), indent=2, default=str)

    elapsed = time.time() - t_start
    total_cost = sum(c.tracker.total_cost for c in clients.values())

    print(f"\n{'='*70}")
    print(f"ARC CRACK v2 RESULTS")
    print(f"{'='*70}")
    print(f"\n  qwen-22b BASELINE: 30.0% (reference from baselines run)")
    print(f"\n  {'Name':50s} {'Rate':>8s} {'Any-OR':>8s} {'Cost':>10s}")
    sorted_results = sorted(all_results.values(), key=lambda x: -x["rate"])
    for r in sorted_results:
        marker = "★" if r["rate"] >= 0.40 else ("●" if r["rate"] >= 0.30 else " ")
        any_str = f"{r.get('any_proposer_rate', r.get('any_correct_rate', 0)):.1%}"
        print(f"  {marker} {r['name']:48s} {r['rate']:>7.1%} {any_str:>8s} ${r['cost']:>8.4f}")

    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    json.dump({
        "experiment": "arc_crack_v2",
        "n_problems": len(tasks),
        "results": all_results,
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only = sys.argv[idx + 1].split(",")
    main(dry_run="--dry-run" in sys.argv, only=only)
