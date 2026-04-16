"""R6: Verifiability gradient — synthesis vs majority vote across 3 verification regimes.

Tests whether the synthesis advantage tracks verification cost:
  1. MMLU-Pro (multiple choice, exact-match verifiable) → "cheap verification"
  2. HumanEval (code, execution-verifiable) → "medium verification" (already done in E9)
  3. ARC-Challenge (science reasoning, no symbolic verifier) → "hard verification"

For each benchmark, we generate K=4 proposals and aggregate two ways:
  (a) Majority vote on extracted answers
  (b) LLM synthesizer reads all K proposals and picks the best

If the synthesis advantage tracks verification cost:
  MMLU-Pro: synth helps (like Phase A math)
  ARC-Challenge: synth hurts or ties (like ARC-AGI visual grids)

This converts the verifiability boundary from a binary observation
(math: helps; grids: hurts) to a gradient across verification regimes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.client import LLMClient


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-half), min(1, center+half))


# ═══════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════

def load_mmlu_pro(n=200, seed=42):
    from datasets import load_dataset
    import random
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    tasks = []
    for i in indices:
        item = ds[i]
        options_text = "\n".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(item["options"]))
        tasks.append({
            "id": f"mmlu_{item['question_id']}",
            "question": item["question"],
            "options": options_text,
            "answer": item["answer"],  # letter like "A", "B", etc.
            "category": item.get("category", ""),
            "type": "multiple_choice",
        })
    return tasks


def load_arc_challenge(n=200, seed=42):
    from datasets import load_dataset
    import random
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    tasks = []
    for i in indices:
        item = ds[i]
        options_text = "\n".join(
            f"({label}) {text}"
            for label, text in zip(item["choices"]["label"], item["choices"]["text"])
        )
        tasks.append({
            "id": f"arc_{item['id']}",
            "question": item["question"],
            "options": options_text,
            "answer": item["answerKey"],
            "category": "science",
            "type": "multiple_choice",
        })
    return tasks


# ═══════════════════════════════════════════════════════════════════
# LLM calls
# ═══════════════════════════════════════════════════════════════════

PROPOSE_SYSTEM = """You are solving a multiple-choice question. Think step by step, then give your final answer as a SINGLE LETTER (A, B, C, D, etc.) on the last line, preceded by "ANSWER:"."""

SYNTH_SYSTEM = """You are given several candidate answers to a multiple-choice question. Read all candidates carefully, evaluate their reasoning, and pick the best answer. Output a SINGLE LETTER (A, B, C, D, etc.) on the last line, preceded by "ANSWER:"."""


def extract_letter(text):
    """Extract a single letter answer from LLM output."""
    if not text:
        return None
    # Look for ANSWER: X
    m = re.search(r'ANSWER:\s*\(?([A-J])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Last single letter on its own line
    for line in reversed(text.strip().split('\n')):
        line = line.strip().strip('().*')
        if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
            return line.upper()
    return None


def run_one_task(client, task, k, synth_temp=0.3):
    """Generate K proposals + 1 synthesizer call for one task."""
    question_prompt = f"{task['question']}\n\n{task['options']}"

    # K proposals at high temperature
    proposals = []
    for si in range(k):
        client.seed = 42 + si
        try:
            resp = client.generate(
                system_prompt=PROPOSE_SYSTEM,
                user_prompt=question_prompt,
                temperature=0.9,
                max_tokens=512,
            )
            letter = extract_letter(resp.content)
            proposals.append({"text": resp.content, "letter": letter, "cost": resp.cost_usd})
        except Exception as e:
            proposals.append({"text": "", "letter": None, "cost": 0})

    # Majority vote
    valid_letters = [p["letter"] for p in proposals if p["letter"]]
    if valid_letters:
        ctr = Counter(valid_letters)
        mv_answer = ctr.most_common(1)[0][0]
    else:
        mv_answer = None

    # LLM synthesizer
    synth_prompt = f"Question:\n{question_prompt}\n\n"
    synth_prompt += "Candidate answers:\n"
    for i, p in enumerate(proposals):
        synth_prompt += f"\n--- Candidate {i+1} ---\n{p['text'][:300]}\n"
    synth_prompt += "\nWhich answer is correct? Evaluate the reasoning and give your final ANSWER: as a single letter."

    client.seed = 999
    try:
        resp = client.generate(
            system_prompt=SYNTH_SYSTEM,
            user_prompt=synth_prompt,
            temperature=synth_temp,
            max_tokens=512,
        )
        synth_answer = extract_letter(resp.content)
        synth_cost = resp.cost_usd
    except Exception:
        synth_answer = None
        synth_cost = 0

    gt = task["answer"]
    total_cost = sum(p["cost"] for p in proposals) + synth_cost

    return {
        "task_id": task["id"],
        "ground_truth": gt,
        "mv_answer": mv_answer,
        "mv_correct": mv_answer == gt,
        "synth_answer": synth_answer,
        "synth_correct": synth_answer == gt,
        "any_correct": any(p["letter"] == gt for p in proposals),
        "cost": total_cost,
    }


def run_benchmark(client, tasks, k, workers, label):
    """Run one benchmark end to end."""
    results = []
    lock = Lock()
    n_done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one_task, client, t, k): t for t in tasks}
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                t = futures[f]
                r = {"task_id": t["id"], "ground_truth": t["answer"],
                     "mv_correct": False, "synth_correct": False, "any_correct": False, "cost": 0}
            with lock:
                results.append(r)
                n_done += 1
                if n_done % 20 == 0:
                    mv_k = sum(1 for x in results if x["mv_correct"])
                    syn_k = sum(1 for x in results if x["synth_correct"])
                    print(f"  [{n_done}/{len(tasks)}] {label}: MV={mv_k}/{n_done} Synth={syn_k}/{n_done}", flush=True)

    elapsed = time.time() - t0
    n = len(results)
    mv_k = sum(1 for r in results if r["mv_correct"])
    syn_k = sum(1 for r in results if r["synth_correct"])
    any_k = sum(1 for r in results if r["any_correct"])
    total_cost = sum(r["cost"] for r in results)
    lo_mv, hi_mv = wilson(mv_k, n)
    lo_s, hi_s = wilson(syn_k, n)

    return {
        "label": label,
        "n": n,
        "majority_vote": mv_k, "mv_rate": mv_k/n,
        "mv_ci": [lo_mv, hi_mv],
        "synthesizer": syn_k, "synth_rate": syn_k/n,
        "synth_ci": [lo_s, hi_s],
        "any_correct": any_k, "any_rate": any_k/n,
        "delta": syn_k - mv_k, "delta_pp": (syn_k - mv_k) / n * 100,
        "cost": total_cost,
        "elapsed_s": elapsed,
        "per_task": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="problems per benchmark")
    p.add_argument("--k", type=int, default=4, help="proposals per problem")
    p.add_argument("--model", default="qwen-22b")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("=" * 70)
    print(f"R6: Verifiability gradient (K={args.k}, n={args.n} per benchmark)")
    print("=" * 70)

    client = LLMClient(args.model, seed=42)

    benchmarks = [
        ("MMLU-Pro (cheap verification)", load_mmlu_pro(args.n)),
        ("ARC-Challenge (hard verification)", load_arc_challenge(args.n)),
    ]

    all_results = {}
    for label, tasks in benchmarks:
        print(f"\n--- {label} ({len(tasks)} problems) ---")
        result = run_benchmark(client, tasks, args.k, args.workers, label)
        all_results[label] = result

        lo_mv, hi_mv = result["mv_ci"]
        lo_s, hi_s = result["synth_ci"]
        print(f"\n  MV:    {result['majority_vote']}/{result['n']} = {result['mv_rate']:.1%}  [{lo_mv:.1%}, {hi_mv:.1%}]")
        print(f"  Synth: {result['synthesizer']}/{result['n']} = {result['synth_rate']:.1%}  [{lo_s:.1%}, {hi_s:.1%}]")
        print(f"  Delta: {result['delta']:+d} ({result['delta_pp']:+.1f}pp)")
        print(f"  Cost:  ${result['cost']:.3f}")

    # Add MATH-500 from R1 (already run)
    math_path = Path("results/alpha_program/r1_math500_topology.json")
    if math_path.exists():
        d = json.load(open(math_path))
        s = d["summary"]
        n = d["n"]
        mv_k = int(s["majority_vote_rate"] * n)
        syn_k = int(s["synth_rate"] * n)
        all_results["MATH-500 (medium verification)"] = {
            "label": "MATH-500 (medium verification)",
            "n": n,
            "majority_vote": mv_k, "mv_rate": s["majority_vote_rate"],
            "synthesizer": syn_k, "synth_rate": s["synth_rate"],
            "delta": syn_k - mv_k,
            "delta_pp": s["delta_synth_minus_mv"] * 100,
        }

    print("\n" + "=" * 70)
    print("R6 FINAL: VERIFIABILITY GRADIENT")
    print("=" * 70)
    print(f"\n{'Benchmark':45s} {'MV':>8s} {'Synth':>8s} {'Delta':>8s} {'Verification'}")
    print("-" * 85)

    gradient_order = [
        "MMLU-Pro (cheap verification)",
        "MATH-500 (medium verification)",
        "ARC-Challenge (hard verification)",
    ]
    for label in gradient_order:
        if label in all_results:
            r = all_results[label]
            print(f"  {label:43s} {r['mv_rate']:>7.1%} {r['synth_rate']:>7.1%} {r['delta_pp']:>+6.1f}pp")

    # Save
    outpath = Path("results/alpha_program") / args.output
    out = {"experiment": "r6_verifiability_gradient", "benchmarks": {}}
    for label, r in all_results.items():
        out["benchmarks"][label] = {k: v for k, v in r.items() if k != "per_task"}
    out["per_task"] = {label: r.get("per_task", []) for label, r in all_results.items()}
    json.dump(out, open(outpath, "w"), indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
