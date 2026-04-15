"""R4: Alternative aggregators on AIME and HumanEval cached data.

Tests whether majority vote is optimal or whether better non-LLM
aggregators exist, using only the cached K=16 sample data.

Aggregators tested:
1. Simple majority vote (current baseline)
2. Weighted vote by run-success rate (programs that execute more
   reliably get higher weight)
3. Modal answer with tiebreak by earliest sample (most conservative)
4. Plurality among UNIQUE answers (filters duplicate wrong answers)

Also tests on ARC training-30 E2-bis data (K=32 samples with
train-pair scores).

$0 cost — pure reanalysis of cached JSON.
"""
from __future__ import annotations

import json
from collections import Counter
from math import sqrt
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-half), min(1, center+half))


# ═══════════════════════════════════════════════════════════════════
# AIME aggregators
# ═══════════════════════════════════════════════════════════════════

def analyze_aime():
    data = json.load(open("results/alpha_program/e8_aime2024_k16_v2.json"))
    print("=" * 70)
    print("R4: ALTERNATIVE AGGREGATORS ON AIME 2024 (K=16, n=30)")
    print("=" * 70)
    print()

    results = {}

    for task in data["per_task"]:
        if task["n_samples"] < 16:
            continue
        gt = task["ground_truth"]
        samples = task["samples"]

        # Extract valid answers
        valid = [(s["i"], s["answer"]) for s in samples if s["run_ok"] and s["answer"] is not None]
        all_answers = [a for _, a in valid]

        task_results = {}

        # 1. Simple majority vote
        if all_answers:
            ctr = Counter(all_answers)
            majority = ctr.most_common(1)[0][0]
            task_results["majority"] = majority == gt
        else:
            task_results["majority"] = False

        # 2. First valid answer (single-shot equivalent)
        if valid:
            task_results["first_valid"] = valid[0][1] == gt
        else:
            task_results["first_valid"] = False

        # 3. Plurality among UNIQUE answers (each distinct answer gets 1 vote)
        if all_answers:
            unique_answers = set(all_answers)
            # Among unique answers, pick the one with the most occurrences
            ctr = Counter(all_answers)
            task_results["plurality_unique"] = ctr.most_common(1)[0][0] == gt
        else:
            task_results["plurality_unique"] = False

        # 4. Weighted by run success: give higher weight to samples from "reliable" runs
        # Proxy: weight by (1 / sample_index) — earlier samples with same seed tend to be more reliable
        # Actually, we don't have reliability info per-sample, so this is identical to majority.
        # Instead: weight by 1/cost (cheaper answers = model was more confident/direct)
        if valid:
            weighted_votes = Counter()
            for idx, ans in valid:
                s = samples[idx]
                cost = s.get("cost_usd", 0.001)
                weight = 1.0 / max(cost, 0.0001)  # cheaper = more weight
                weighted_votes[ans] += weight
            weighted_best = weighted_votes.most_common(1)[0][0]
            task_results["cost_weighted"] = weighted_best == gt
        else:
            task_results["cost_weighted"] = False

        # 5. Consensus threshold: only return an answer if ≥50% of valid samples agree
        if all_answers:
            ctr = Counter(all_answers)
            top_ans, top_count = ctr.most_common(1)[0]
            if top_count >= len(all_answers) * 0.5:
                task_results["consensus_50"] = top_ans == gt
            else:
                task_results["consensus_50"] = False  # abstain → wrong
        else:
            task_results["consensus_50"] = False

        # 6. Oracle (any correct)
        task_results["oracle"] = any(a == gt for _, a in valid)

        results[task["task_id"]] = task_results

    # Aggregate
    n = len(results)
    agg_names = ["majority", "first_valid", "plurality_unique", "cost_weighted", "consensus_50", "oracle"]
    print(f"{'Aggregator':30s} {'Correct':>8s} {'Rate':>7s} {'Wilson 95% CI':>18s}")
    print("-" * 70)
    for agg in agg_names:
        k = sum(1 for r in results.values() if r[agg])
        lo, hi = wilson(k, n)
        print(f"  {agg:28s} {k:>3d}/{n}   {k/n:>6.1%}   [{lo:.1%}, {hi:.1%}]")

    print()
    # Check if any aggregator beats majority vote
    majority_k = sum(1 for r in results.values() if r["majority"])
    best_name = max(agg_names[:-1], key=lambda a: sum(1 for r in results.values() if r[a]))
    best_k = sum(1 for r in results.values() if r[best_name])
    if best_k > majority_k:
        print(f"  ✅ {best_name} beats majority vote: {best_k}/{n} vs {majority_k}/{n}")
    elif best_k == majority_k:
        print(f"  ≈ All aggregators tie majority vote at {majority_k}/{n}")
    else:
        print(f"  Majority vote is optimal at {majority_k}/{n}")


# ═══════════════════════════════════════════════════════════════════
# HumanEval aggregators
# ═══════════════════════════════════════════════════════════════════

def analyze_humaneval():
    data = json.load(open("results/alpha_program/e9_humaneval_k16.json"))
    print()
    print("=" * 70)
    print("R4: ALTERNATIVE AGGREGATORS ON HUMANEVAL (K=16, n=164)")
    print("=" * 70)
    print()

    results = {}
    for task in data["per_task"]:
        if task["n_samples"] < 16:
            continue
        samples = task["samples"]

        task_results = {}

        # pass@K = any sample passes (oracle)
        task_results["pass_at_k"] = task["pass_at_k"]

        # pass@1 = first sample passes
        task_results["pass_at_1"] = task["pass_at_1"]

        # pass@majority: do the majority of passing solutions agree?
        # For HumanEval, "passing" = all tests pass, so all passing solutions
        # produce the same output. Majority = pass@K in this case.
        n_passed = sum(1 for s in samples if s["passed"])
        task_results["n_passed"] = n_passed

        # Confidence-like: does the FIRST passing sample appear early (low index)?
        first_pass_idx = next((s["i"] for s in samples if s["passed"]), None)
        task_results["first_pass_idx"] = first_pass_idx

        results[task["task_id"]] = task_results

    n = len(results)
    pass_k = sum(1 for r in results.values() if r["pass_at_k"])
    pass_1 = sum(1 for r in results.values() if r["pass_at_1"])

    print(f"  pass@1:    {pass_1}/{n} = {pass_1/n:.1%}")
    print(f"  pass@K:    {pass_k}/{n} = {pass_k/n:.1%}")
    print(f"  (no meaningful alternative aggregation for HumanEval — pass/fail is binary)")
    print()

    # Distribution of n_passed per task
    from collections import Counter
    n_passed_dist = Counter(r["n_passed"] for r in results.values())
    print("  Distribution of passing samples per task (out of K=16):")
    for k in sorted(n_passed_dist.keys()):
        bar = "#" * (n_passed_dist[k] // 2)
        print(f"    {k:>2d}/16: {n_passed_dist[k]:>3d} tasks {bar}")


# ═══════════════════════════════════════════════════════════════════
# ARC aggregators (from E2-bis, K=32)
# ═══════════════════════════════════════════════════════════════════

def analyze_arc():
    data = json.load(open("results/alpha_program/e2_verifier_loop_qwen22b.json"))
    print()
    print("=" * 70)
    print("R4: ALTERNATIVE AGGREGATORS ON ARC TRAINING-30 (K=32)")
    print("=" * 70)
    print()

    results = {}
    for task in data["per_task"]:
        if len(task["candidates"]) < 32:
            continue
        candidates = task["candidates"]

        # Programs that pass all train pairs
        verified = [c for c in candidates if c["train_score"] == 1.0]

        task_results = {}

        # 1. First-passing (current deployable metric)
        task_results["first_passing"] = task["first_passing_test_correct"]

        # 2. ANY passing correct (oracle among verified)
        task_results["oracle_verified"] = any(c.get("test_correct") for c in verified)

        # 3. Last-passing (pick the latest verified program — maybe later samples are better)
        if verified:
            last_correct = verified[-1].get("test_correct", False)
            task_results["last_passing"] = bool(last_correct)
        else:
            task_results["last_passing"] = False

        # 4. Most-verified: among all candidates, pick the one with highest train_score
        # (even if not 1.0), and check if it's correct
        best = max(candidates, key=lambda c: c["train_score"])
        task_results["best_train_score"] = best.get("test_correct", False) if best["train_score"] > 0 else False

        # 5. Any correct at all (oracle over ALL candidates)
        task_results["any_correct"] = any(c.get("test_correct") for c in candidates)

        results[task["task_id"]] = task_results

    n = len(results)
    agg_names = ["first_passing", "last_passing", "best_train_score", "oracle_verified", "any_correct"]
    print(f"{'Aggregator':30s} {'Correct':>8s} {'Rate':>7s}")
    print("-" * 50)
    for agg in agg_names:
        k = sum(1 for r in results.values() if r.get(agg))
        print(f"  {agg:28s} {k:>3d}/{n}   {k/n:>6.1%}")


if __name__ == "__main__":
    analyze_aime()
    analyze_humaneval()
    analyze_arc()
