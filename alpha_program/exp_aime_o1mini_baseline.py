"""Same-task o1-mini baseline on AIME 2024 — bulletproofing the 278× claim.

Runs o1-mini (OpenAI) on the exact same 30 AIME 2024 problems our
architecture ran on, with the same grading (integer match). This
provides a same-task, same-evaluation-condition comparison instead of
relying on published numbers from potentially different conditions.

AIME answers are integers in [0, 999]. We ask o1-mini to solve each
problem and extract the integer answer from its response.

Usage:
    python -m alpha_program.exp_aime_o1mini_baseline --output aime_o1mini_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.aime_tasks import AIMETask, load_aime_tasks
from src.config import RESULTS_DIR

from openai import OpenAI


def extract_integer_answer(text: str) -> int | None:
    """Extract an integer answer from o1-mini's response."""
    if not text:
        return None
    # Look for common patterns: "the answer is 42", "= 42", "**42**", last number in text
    # Try "answer is X" pattern first
    m = re.search(r'(?:answer|result|value)\s*(?:is|=|:)\s*[\\$]*\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Try boxed answer (LaTeX): \boxed{42}
    m = re.search(r'\\boxed\{(\d+)\}', text)
    if m:
        return int(m.group(1))
    # Try bold: **42**
    m = re.search(r'\*\*(\d+)\*\*\s*$', text.strip(), re.MULTILINE)
    if m:
        return int(m.group(1))
    # Last resort: last integer in the text
    all_ints = re.findall(r'\b(\d+)\b', text)
    if all_ints:
        val = int(all_ints[-1])
        if 0 <= val <= 999:
            return val
    return None


@dataclass
class Result:
    task_id: str
    ground_truth: int
    model_answer: int | None
    correct: bool
    cost_usd: float
    tokens_in: int
    tokens_out: int
    response_text: str
    error: str | None = None


def run_one(client: OpenAI, task: AIMETask, model: str, max_tokens: int) -> Result:
    prompt = (
        f"Solve this AIME problem. The answer is an integer from 0 to 999.\n\n"
        f"{task.problem}\n\n"
        f"Think step by step, then give your final answer as a single integer."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        # o1-mini pricing: $1.10/M input, $4.40/M output (as of 2026-04)
        cost = (tokens_in * 1.10 + tokens_out * 4.40) / 1_000_000
        answer = extract_integer_answer(content)
        correct = answer == task.answer
        return Result(
            task_id=task.id, ground_truth=task.answer,
            model_answer=answer, correct=correct,
            cost_usd=cost, tokens_in=tokens_in, tokens_out=tokens_out,
            response_text=content,
        )
    except Exception as e:
        return Result(
            task_id=task.id, ground_truth=task.answer,
            model_answer=None, correct=False,
            cost_usd=0, tokens_in=0, tokens_out=0,
            response_text="", error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-half), min(1, center+half))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="o1-mini")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("=" * 70)
    print(f"Same-task AIME 2024 baseline: {args.model}")
    print("=" * 70)

    tasks = load_aime_tasks(year=2024, n=-1, seed=42)
    print(f"Loaded {len(tasks)} AIME 2024 problems (same as our architecture)")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    results: list[Result] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, client, t, args.model, args.max_tokens): t for t in tasks}
        for f in as_completed(futures):
            t = futures[f]
            try:
                r = f.result()
            except Exception as e:
                r = Result(task_id=t.id, ground_truth=t.answer, model_answer=None,
                          correct=False, cost_usd=0, tokens_in=0, tokens_out=0,
                          response_text="", error=str(e))
            results.append(r)
            n_correct = sum(1 for x in results if x.correct)
            mark = "✓" if r.correct else "·"
            print(f"  [{len(results)}/{len(tasks)}] {r.task_id} {mark} "
                  f"answer={r.model_answer} (gt={r.ground_truth}) "
                  f"running={n_correct}/{len(results)} "
                  f"cost=${sum(x.cost_usd for x in results):.3f}", flush=True)

    elapsed = time.time() - t0
    n = len(results)
    n_correct = sum(1 for r in results if r.correct)
    total_cost = sum(r.cost_usd for r in results)
    lo, hi = wilson(n_correct, n)

    print()
    print("=" * 70)
    print(f"FINAL — {args.model} on AIME 2024 (same-task baseline)")
    print("=" * 70)
    print(f"  Correct: {n_correct}/{n} = {n_correct/n:.1%}  Wilson [{lo:.1%}, {hi:.1%}]")
    print(f"  Cost: ${total_cost:.4f} total, ${total_cost/n:.4f}/task")
    if n_correct > 0:
        cpc = (total_cost / n) / (n_correct / n)
        print(f"  Cost-per-correct: ${cpc:.4f}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print()
    print("Same-task comparison:")
    print(f"  AlphaProgram (qwen-22b K=16): 23/30 = 76.7%, $0.005/task, $0.0065/correct")
    print(f"  {args.model} (this run):       {n_correct}/{n} = {n_correct/n:.1%}, ${total_cost/n:.4f}/task, ${(total_cost/n)/(n_correct/n) if n_correct else float('inf'):.4f}/correct")
    if n_correct > 0:
        cost_ratio = ((total_cost/n)/(n_correct/n)) / 0.0065
        print(f"  Cost-per-correct ratio: {cost_ratio:.0f}×")

    # Save
    out = {
        "experiment": "aime_same_task_baseline",
        "model": args.model,
        "n": n,
        "n_correct": n_correct,
        "rate": n_correct / n,
        "total_cost_usd": total_cost,
        "cost_per_task": total_cost / n,
        "elapsed_s": elapsed,
        "per_task": [
            {
                "task_id": r.task_id,
                "ground_truth": r.ground_truth,
                "model_answer": r.model_answer,
                "correct": r.correct,
                "cost_usd": r.cost_usd,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "error": r.error,
            }
            for r in results
        ],
    }
    json.dump(out, open(output_path, "w"), indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
