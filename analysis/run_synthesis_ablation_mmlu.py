"""Synthesis ablation on MMLU-Pro — 3-way decomposition on a non-math benchmark.

Same design as the MATH-500 ablation:
  (A) Majority vote on extracted letter answers
  (B) LLM synth reading only the final letter answers
  (C) LLM synth reading full reasoning traces

Tests whether the synthesis decomposition is math-specific or general.
"""
from __future__ import annotations

import argparse
import json
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


def load_mmlu_pro(n=500, seed=42):
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
            "question": item["question"],
            "options": options_text,
            "answer": item["answer"],
        })
    return tasks


def extract_letter(text):
    if not text:
        return None
    m = re.search(r'ANSWER:\s*\(?([A-J])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for line in reversed(text.strip().split('\n')):
        line = line.strip().strip('().*')
        if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
            return line.upper()
    return None


PROPOSE_SYSTEM = """You are solving a multiple-choice question. Think step by step, then give your final answer as a SINGLE LETTER preceded by "ANSWER:"."""

SYNTH_ANS_SYSTEM = """You are given several candidate answers (letters only) to a multiple-choice question. Pick the one most likely to be correct. Output "ANSWER:" followed by a single letter."""

SYNTH_FULL_SYSTEM = """You are given several candidate solutions with full reasoning. Evaluate each, then pick the best answer. Output "ANSWER:" followed by a single letter."""


def run_one(client, task, k):
    question_prompt = f"{task['question']}\n\n{task['options']}"
    gt = task["answer"]

    proposals = []
    for si in range(k):
        client.seed = 42 + si
        try:
            resp = client.generate(PROPOSE_SYSTEM, question_prompt, temperature=0.9, max_tokens=512)
            letter = extract_letter(resp.content)
            proposals.append({"text": resp.content, "letter": letter, "cost": resp.cost_usd})
        except:
            proposals.append({"text": "", "letter": None, "cost": 0})

    valid = [p["letter"] for p in proposals if p["letter"]]
    mv_answer = Counter(valid).most_common(1)[0][0] if valid else None
    mv_correct = mv_answer == gt

    # Synth answers only
    ans_text = "\n".join(f"Candidate {i+1}: {p['letter'] or '?'}" for i, p in enumerate(proposals))
    client.seed = 900
    try:
        resp = client.generate(SYNTH_ANS_SYSTEM, f"Question:\n{question_prompt}\n\nCandidate answers:\n{ans_text}", temperature=0.3, max_tokens=128)
        sa_answer = extract_letter(resp.content)
    except:
        sa_answer = None
    sa_correct = sa_answer == gt

    # Synth full reasoning
    full_text = "\n\n".join(f"--- Candidate {i+1} ---\n{p['text'][:300]}" for i, p in enumerate(proposals))
    client.seed = 901
    try:
        resp = client.generate(SYNTH_FULL_SYSTEM, f"Question:\n{question_prompt}\n\n{full_text}", temperature=0.3, max_tokens=256)
        sf_answer = extract_letter(resp.content)
    except:
        sf_answer = None
    sf_correct = sf_answer == gt

    return {"mv": mv_correct, "sa": sa_correct, "sf": sf_correct}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--model", default="qwen-22b")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Synthesis ablation on MMLU-Pro (model={args.model}, K={args.k}, n={args.n})")
    tasks = load_mmlu_pro(args.n)
    print(f"Loaded {len(tasks)} MMLU-Pro problems")
    client = LLMClient(args.model, seed=42)

    results = []
    lock = Lock()
    n_done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, client, t, args.k): i for i, t in enumerate(tasks)}
        for f in as_completed(futures):
            try:
                r = f.result()
            except:
                r = {"mv": False, "sa": False, "sf": False}
            with lock:
                results.append(r)
                n_done += 1
                if n_done % 50 == 0:
                    mv = sum(1 for x in results if x["mv"])
                    sa = sum(1 for x in results if x["sa"])
                    sf = sum(1 for x in results if x["sf"])
                    print(f"  [{n_done}/{len(tasks)}] MV={mv} SA={sa} SF={sf}", flush=True)

    n = len(results)
    mv_k = sum(1 for r in results if r["mv"])
    sa_k = sum(1 for r in results if r["sa"])
    sf_k = sum(1 for r in results if r["sf"])

    lo_mv, hi_mv = wilson(mv_k, n)
    lo_sa, hi_sa = wilson(sa_k, n)
    lo_sf, hi_sf = wilson(sf_k, n)

    print(f"\n{'='*60}")
    print(f"SYNTHESIS ABLATION — MMLU-Pro ({args.model})")
    print(f"{'='*60}")
    print(f"  (A) Majority vote:        {mv_k}/{n} = {mv_k/n:.1%}  [{lo_mv:.1%}, {hi_mv:.1%}]")
    print(f"  (B) Synth (answers only): {sa_k}/{n} = {sa_k/n:.1%}  [{lo_sa:.1%}, {hi_sa:.1%}]  Δ={(sa_k-mv_k)/n*100:+.1f}pp")
    print(f"  (C) Synth (full reason):  {sf_k}/{n} = {sf_k/n:.1%}  [{lo_sf:.1%}, {hi_sf:.1%}]  Δ={(sf_k-mv_k)/n*100:+.1f}pp")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    out = {
        "experiment": "synthesis_ablation_mmlu_pro",
        "model": args.model, "n": n, "k": args.k,
        "majority_vote": {"k": mv_k, "rate": mv_k/n, "ci": [lo_mv, hi_mv]},
        "synth_answers_only": {"k": sa_k, "rate": sa_k/n, "ci": [lo_sa, hi_sa]},
        "synth_full_reasoning": {"k": sf_k, "rate": sf_k/n, "ci": [lo_sf, hi_sf]},
        "elapsed_s": time.time()-t0,
    }
    outpath = Path("results/alpha_program") / args.output
    json.dump(out, open(outpath, "w"), indent=2, default=str)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
