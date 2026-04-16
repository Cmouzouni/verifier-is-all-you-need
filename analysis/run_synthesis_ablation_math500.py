"""Synthesis ablation on MATH-500 — external validation of the +14pp/+5pp decomposition.

On Phase A, the synthesis ablation showed:
  Majority vote: 70.8%
  LLM synth (answers only): 84.8% (+14.0pp)
  LLM synth (full reasoning): 89.6% (+4.8pp more)

This experiment replicates the same decomposition on MATH-500:
  (A) Generate K=4 proposals per problem
  (B) Aggregate three ways:
      1. Majority vote on extracted answers
      2. LLM synth that sees only the final answers
      3. LLM synth that sees full reasoning traces
  (C) Compare correctness rates

If the decomposition replicates: LLM aggregation is the active ingredient
on standard math too, confirming the Phase A mechanism.
If it doesn't: the +14pp was specific to Phase A's hand-crafted structure.

Uses qwen-7b for speed (~1.3 calls/sec vs 0.23 for qwen-22b).
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


def load_math(n=500, seed=42):
    from datasets import load_dataset
    import random
    # Load all MATH subjects and combine
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    all_items = []
    for subj in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        all_items.extend(list(ds))
    ds = all_items
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in indices]


def extract_boxed(text):
    """Extract \\boxed{...} answer from MATH solution."""
    # Find the LAST \boxed{}
    matches = list(re.finditer(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_answer(text):
    """Extract a numeric/symbolic answer from model output."""
    if not text:
        return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Try \boxed first
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    # Try ANSWER: pattern
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Last line that looks like an answer
    for line in reversed(text.strip().split('\n')):
        line = line.strip().rstrip('.')
        if line and len(line) < 50 and not line.startswith(('The', 'So', 'Thus', 'Therefore', 'Hence', 'We')):
            return line
    return None


def normalize_answer(ans):
    """Normalize for comparison."""
    if ans is None:
        return None
    ans = str(ans).strip()
    ans = ans.replace('\\$', '').replace('$', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\,', '').replace('\\ ', '')
    ans = ans.strip()
    # Try numeric
    try:
        return str(float(ans))
    except:
        pass
    return ans.lower()


def answers_match(pred, gold):
    """Check if predicted answer matches gold."""
    if pred is None or gold is None:
        return False
    np, ng = normalize_answer(pred), normalize_answer(gold)
    if np == ng:
        return True
    # Try sympy for symbolic equivalence
    try:
        from sympy import simplify, sympify
        diff = simplify(sympify(np) - sympify(ng))
        return diff == 0
    except:
        pass
    return False


PROPOSE_SYSTEM = """You are solving a math competition problem. Think step by step, show your work, then give your final answer inside \\boxed{}."""

SYNTH_ANSWERS_SYSTEM = """You are given several candidate answers to a math problem. You do NOT see the reasoning — only the final answers. Pick the answer you believe is most likely correct. Output your chosen answer inside \\boxed{}."""

SYNTH_FULL_SYSTEM = """You are given several candidate solutions to a math problem, including their full reasoning. Read each carefully, evaluate the reasoning, and pick the best answer. Output your chosen answer inside \\boxed{}."""


def run_one_problem(client, problem, k, synth_model_key=None):
    """Generate K proposals and aggregate 3 ways."""
    question = problem["problem"]
    gold_answer = extract_boxed(problem["solution"]) or ""

    # Generate K proposals
    proposals = []
    for si in range(k):
        client.seed = 42 + si
        try:
            resp = client.generate(PROPOSE_SYSTEM, question, temperature=0.9, max_tokens=1024)
            answer = extract_answer(resp.content)
            proposals.append({"text": resp.content, "answer": answer, "cost": resp.cost_usd})
        except:
            proposals.append({"text": "", "answer": None, "cost": 0})

    # Aggregation 1: Majority vote
    valid = [normalize_answer(p["answer"]) for p in proposals if p["answer"]]
    if valid:
        ctr = Counter(valid)
        mv_answer = ctr.most_common(1)[0][0]
    else:
        mv_answer = None
    mv_correct = answers_match(mv_answer, gold_answer)

    # Aggregation 2: Synth (answers only)
    answers_text = "\n".join(f"Candidate {i+1}: {p['answer'] or '(no answer)'}" for i, p in enumerate(proposals))
    synth_prompt = f"Problem:\n{question}\n\nCandidate answers:\n{answers_text}\n\nWhich answer is correct?"
    client.seed = 900
    try:
        resp = client.generate(SYNTH_ANSWERS_SYSTEM, synth_prompt, temperature=0.3, max_tokens=256)
        synth_ans_answer = extract_answer(resp.content)
        synth_ans_cost = resp.cost_usd
    except:
        synth_ans_answer = None
        synth_ans_cost = 0
    synth_ans_correct = answers_match(synth_ans_answer, gold_answer)

    # Aggregation 3: Synth (full reasoning)
    full_text = "\n\n".join(f"--- Candidate {i+1} ---\n{p['text'][:400]}" for i, p in enumerate(proposals))
    synth_full_prompt = f"Problem:\n{question}\n\n{full_text}\n\nWhich solution is correct? Give the answer in \\boxed{{}}."
    client.seed = 901
    try:
        resp = client.generate(SYNTH_FULL_SYSTEM, synth_full_prompt, temperature=0.3, max_tokens=256)
        synth_full_answer = extract_answer(resp.content)
        synth_full_cost = resp.cost_usd
    except:
        synth_full_answer = None
        synth_full_cost = 0
    synth_full_correct = answers_match(synth_full_answer, gold_answer)

    total_cost = sum(p["cost"] for p in proposals) + synth_ans_cost + synth_full_cost

    return {
        "gold": gold_answer,
        "mv_correct": mv_correct,
        "synth_ans_correct": synth_ans_correct,
        "synth_full_correct": synth_full_correct,
        "cost": total_cost,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--model", default="qwen-7b")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Synthesis ablation on MATH-500 (model={args.model}, K={args.k}, n={args.n})")
    problems = load_math(args.n)
    print(f"Loaded {len(problems)} MATH problems")
    client = LLMClient(args.model, seed=42)

    results = []
    lock = Lock()
    n_done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_problem, client, prob, args.k): i for i, prob in enumerate(problems)}
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                r = {"mv_correct": False, "synth_ans_correct": False, "synth_full_correct": False, "cost": 0}
            with lock:
                results.append(r)
                n_done += 1
                if n_done % 50 == 0:
                    mv = sum(1 for x in results if x["mv_correct"])
                    sa = sum(1 for x in results if x["synth_ans_correct"])
                    sf = sum(1 for x in results if x["synth_full_correct"])
                    print(f"  [{n_done}/{len(problems)}] MV={mv}/{n_done} SynthAns={sa}/{n_done} SynthFull={sf}/{n_done}", flush=True)

    elapsed = time.time() - t0
    n = len(results)
    mv_k = sum(1 for r in results if r["mv_correct"])
    sa_k = sum(1 for r in results if r["synth_ans_correct"])
    sf_k = sum(1 for r in results if r["synth_full_correct"])
    cost = sum(r["cost"] for r in results)

    lo_mv, hi_mv = wilson(mv_k, n)
    lo_sa, hi_sa = wilson(sa_k, n)
    lo_sf, hi_sf = wilson(sf_k, n)

    print(f"\n{'='*60}")
    print(f"SYNTHESIS ABLATION — MATH-500 ({args.model})")
    print(f"{'='*60}")
    print(f"  (A) Majority vote:        {mv_k}/{n} = {mv_k/n:.1%}  [{lo_mv:.1%}, {hi_mv:.1%}]")
    print(f"  (B) Synth (answers only): {sa_k}/{n} = {sa_k/n:.1%}  [{lo_sa:.1%}, {hi_sa:.1%}]  Δ={sa_k/n - mv_k/n:+.1%}")
    print(f"  (C) Synth (full reason):  {sf_k}/{n} = {sf_k/n:.1%}  [{lo_sf:.1%}, {hi_sf:.1%}]  Δ={sf_k/n - mv_k/n:+.1%}")
    print(f"  Cost: ${cost:.3f}, Elapsed: {elapsed:.0f}s")
    print(f"\n  Phase A reference: MV 70.8% → Synth-Ans 84.8% (+14pp) → Synth-Full 89.6% (+5pp)")
    print(f"  MATH-500 result:   MV {mv_k/n:.1%} → Synth-Ans {sa_k/n:.1%} ({(sa_k-mv_k)/n*100:+.1f}pp) → Synth-Full {sf_k/n:.1%} ({(sf_k-mv_k)/n*100:+.1f}pp)")

    out = {
        "experiment": "synthesis_ablation_math500",
        "model": args.model, "n": n, "k": args.k,
        "majority_vote": {"k": mv_k, "rate": mv_k/n, "ci": [lo_mv, hi_mv]},
        "synth_answers_only": {"k": sa_k, "rate": sa_k/n, "ci": [lo_sa, hi_sa]},
        "synth_full_reasoning": {"k": sf_k, "rate": sf_k/n, "ci": [lo_sf, hi_sf]},
        "cost": cost, "elapsed_s": elapsed,
    }
    outpath = Path("results/alpha_program") / args.output
    json.dump(out, open(outpath, "w"), indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
