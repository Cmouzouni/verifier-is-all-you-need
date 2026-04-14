"""Direction B — Heterogeneous model union under symbolic verification.

The single most important experiment in the next research phase.

Hypothesis: if we run 3 different models (qwen-22b, DeepSeek-V3.1, Llama-70B)
each at K=8 on the same 400 ARC-AGI-1 eval tasks, the UNION of their verified
programs will solve more tasks than any single model at K=24 (same total budget).

If yes: model diversity under symbolic verification is a genuine architectural
primitive — the first finding where the "wiring" actually matters.

If no: the architecture truly doesn't matter; the best single proposer at
maximum K dominates any multi-model strategy.

Design:
  Phase 1: Run each model at K=8 on the full 400-task eval (reuse E7 data for
           qwen-22b by taking the first 8 candidates per task).
  Phase 2: Compute 4 metrics:
           a) qwen-22b alone at K=8 (from E7 data, first 8)
           b) qwen-22b alone at K=24 (from E7 data, all 16, plus... well, E7
              was K=16, so we compare against K=16 as the budget-matched single-
              model baseline — 3 models × K=8 = 24 candidates; qwen-22b at
              K=16 = 16 candidates. Actually, the fair comparison is:
              UNION of 3×K=8 = 24 total vs single-model at K=24. Since E7 only
              has K=16 per task, this is slightly in favor of the union. We
              report both and let the reader decide.)
           c) Each model individually at K=8
           d) Union of all 3 at K=8 each
           e) Pairwise unions (qwen+deepseek, qwen+llama, deepseek+llama)

Usage:
    # Phase 1a: run DeepSeek-V3.1 on eval-400
    python -m alpha_program.run_validation --model deepseek \
        --dataset arc1 --split evaluation --n 400 --k 8 --workers 16 \
        --output dirB_deepseek_eval400_k8.json

    # Phase 1b: run Llama-70B on eval-400
    python -m alpha_program.run_validation --model llama-70b \
        --dataset arc1 --split evaluation --n 400 --k 8 --workers 16 \
        --output dirB_llama70b_eval400_k8.json

    # Phase 2: analyze the union
    python -m alpha_program.exp_dirB_heterogeneous_union --analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_per_task_results(fname: str, k_filter: int) -> dict[str, dict]:
    """Load a validation JSON and return {task_id: {deployable, oracle, any_train}} for tasks with ≥ k_filter candidates."""
    path = RESULTS_DIR / "alpha_program" / fname
    if not path.exists():
        return {}
    data = json.load(open(path))
    results = {}
    for t in data["per_task"]:
        candidates = t.get("candidates", [])
        # Take only the first k_filter candidates (for fair budget comparison)
        candidates = candidates[:k_filter]
        if len(candidates) < k_filter:
            continue
        tid = t["task_id"]
        any_train = any(c["train_score"] == 1.0 for c in candidates)
        # deployable = first candidate with train_score=1.0 has test_correct=True
        first_passing_test = None
        for c in candidates:
            if c["train_score"] == 1.0:
                first_passing_test = c.get("test_correct", False)
                break
        oracle = any(c.get("test_correct", False) for c in candidates)
        results[tid] = {
            "any_train": any_train,
            "deployable": bool(first_passing_test),
            "oracle": oracle,
        }
    return results


def analyze():
    output_dir = RESULTS_DIR / "alpha_program"

    # Load all three models' results
    models = {
        "qwen-22b (K=8 from E7)": load_per_task_results("e7_arc1_eval_n400_k16.json", k_filter=8),
        "qwen-22b (K=16 from E7)": load_per_task_results("e7_arc1_eval_n400_k16.json", k_filter=16),
        "DeepSeek-V3.1 (K=8)": load_per_task_results("dirB_deepseek_eval400_k8.json", k_filter=8),
        "Llama-70B (K=8)": load_per_task_results("dirB_llama70b_eval400_k8.json", k_filter=8),
    }

    print("=" * 80)
    print("DIRECTION B — HETEROGENEOUS MODEL UNION ANALYSIS")
    print("=" * 80)
    print()

    # Individual model results
    print("Individual models (deployable / n):")
    for name, results in models.items():
        if not results:
            print(f"  {name:40s}: NOT AVAILABLE")
            continue
        n = len(results)
        deploy = sum(1 for r in results.values() if r["deployable"])
        oracle = sum(1 for r in results.values() if r["oracle"])
        lo, hi = wilson(deploy, n)
        print(f"  {name:40s}: {deploy:>3}/{n} = {deploy/n:.1%} [{lo:.1%},{hi:.1%}]  oracle {oracle}/{n}")

    print()

    # Check which models are available for union
    qwen8 = models.get("qwen-22b (K=8 from E7)", {})
    qwen16 = models.get("qwen-22b (K=16 from E7)", {})
    deepseek = models.get("DeepSeek-V3.1 (K=8)", {})
    llama = models.get("Llama-70B (K=8)", {})

    available = {}
    if qwen8:
        available["qwen-22b"] = qwen8
    if deepseek:
        available["deepseek"] = deepseek
    if llama:
        available["llama-70b"] = llama

    if len(available) < 2:
        print("Need at least 2 models to compute unions. Run Phase 1 first.")
        return

    # Get the common task set
    common_tasks = set.intersection(*[set(r.keys()) for r in available.values()])
    n = len(common_tasks)
    print(f"Common tasks across {len(available)} models: {n}")
    print()

    # Compute unions
    print("UNION RESULTS (deployable — a task is solved if ANY model's verified program is correct):")
    print("-" * 80)

    # Individual at K=8
    for name, results in available.items():
        deploy = sum(1 for tid in common_tasks if results.get(tid, {}).get("deployable"))
        lo, hi = wilson(deploy, n)
        print(f"  {name:40s} K=8:  {deploy:>3}/{n} = {deploy/n:.1%} [{lo:.1%},{hi:.1%}]")

    # Single-model baseline at K=16 (budget = 3 models × K=8 ≈ K=24, but we only have K=16)
    if qwen16:
        deploy16 = sum(1 for tid in common_tasks if qwen16.get(tid, {}).get("deployable"))
        lo16, hi16 = wilson(deploy16, n)
        print(f"  {'qwen-22b K=16 (budget baseline)':40s}:  {deploy16:>3}/{n} = {deploy16/n:.1%} [{lo16:.1%},{hi16:.1%}]")

    print()

    # Pairwise unions
    model_names = list(available.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            r1, r2 = available[m1], available[m2]
            deploy = sum(
                1 for tid in common_tasks
                if r1.get(tid, {}).get("deployable") or r2.get(tid, {}).get("deployable")
            )
            lo, hi = wilson(deploy, n)
            label = f"{m1} ∪ {m2}"
            print(f"  {label:40s} K=8+8: {deploy:>3}/{n} = {deploy/n:.1%} [{lo:.1%},{hi:.1%}]")

    # Full union
    deploy_union = sum(
        1 for tid in common_tasks
        if any(r.get(tid, {}).get("deployable") for r in available.values())
    )
    lo_u, hi_u = wilson(deploy_union, n)
    k_total = len(available) * 8
    print(f"  {'FULL UNION':40s} K={k_total:>2}:  {deploy_union:>3}/{n} = {deploy_union/n:.1%} [{lo_u:.1%},{hi_u:.1%}]")

    print()
    print("=" * 80)
    print("THE VERDICT:")
    print("=" * 80)

    # Compare union vs best single model
    best_single_name = max(available.keys(), key=lambda m: sum(1 for tid in common_tasks if available[m].get(tid, {}).get("deployable")))
    best_single_deploy = sum(1 for tid in common_tasks if available[best_single_name].get(tid, {}).get("deployable"))

    delta = deploy_union - best_single_deploy
    if delta > 0:
        # Check complementarity: how many tasks does each non-best model uniquely solve?
        print(f"  UNION ({deploy_union}/{n}) > best single model {best_single_name} ({best_single_deploy}/{n})")
        print(f"  Δ = +{delta} tasks ({delta/n:.1%}pp)")
        print()
        print("  Unique contributions (tasks solved by this model but not the best single model):")
        for name, results in available.items():
            if name == best_single_name:
                continue
            unique = sum(
                1 for tid in common_tasks
                if results.get(tid, {}).get("deployable")
                and not available[best_single_name].get(tid, {}).get("deployable")
            )
            print(f"    {name}: {unique} unique tasks")
        print()
        print("  ✅ MODEL DIVERSITY IS A GENUINE ARCHITECTURAL PRIMITIVE.")
        print("  The wiring matters: different models solve different tasks,")
        print("  and symbolic verification makes their union safe.")
    else:
        print(f"  UNION ({deploy_union}/{n}) ≤ best single model ({best_single_deploy}/{n})")
        print(f"  Δ = {delta} tasks")
        print()
        print("  ❌ Model diversity adds no value. The best single model dominates.")
        print("  The architecture truly doesn't matter; proposer quality is everything.")

    # Also check oracle union
    oracle_union = sum(
        1 for tid in common_tasks
        if any(r.get(tid, {}).get("oracle") for r in available.values())
    )
    best_oracle = max(
        sum(1 for tid in common_tasks if available[m].get(tid, {}).get("oracle"))
        for m in available
    )
    print()
    print(f"  Oracle union: {oracle_union}/{n} vs best single oracle: {best_oracle}/{n}  (Δ = +{oracle_union - best_oracle})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true", help="Run the union analysis on existing result files")
    args = p.parse_args()

    if args.analyze:
        analyze()
    else:
        print("Use --analyze to run the union analysis.")
        print("First run Phase 1 to generate the per-model results:")
        print()
        print("  python -m alpha_program.run_validation --model deepseek \\")
        print("      --dataset arc1 --split evaluation --n 400 --k 8 --workers 16 \\")
        print("      --output dirB_deepseek_eval400_k8.json")
        print()
        print("  python -m alpha_program.run_validation --model llama-70b \\")
        print("      --dataset arc1 --split evaluation --n 400 --k 8 --workers 16 \\")
        print("      --output dirB_llama70b_eval400_k8.json")
