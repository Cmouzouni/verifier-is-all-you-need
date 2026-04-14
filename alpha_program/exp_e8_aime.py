"""E8 — AlphaProgram on AIME 2024 with code execution + self-consistency.

Each AIME problem is fed to qwen-22b K times with different sampling
seeds. Each sample is asked to write `def solve()` that returns the
integer answer. We sandbox-execute each solve(); valid integer answers
are voted via majority vote across the K samples. Final answer = the
most common integer.

Compared to the ARC verifier-loop architecture, the only structural
difference is that AIME has no train pairs to verify against — selection
is by majority vote on the executed answers, not by passing a held-out
filter. This is the program-of-thought + self-consistency template.

Pass criterion: cost-per-correct on AIME 2024 better than published
frontier reasoners. Target: ~30-50% accuracy at $0.05/task ⇒
$0.10-0.16 per correct, vs o1 ~83% at $1.50/task = $1.81 per correct.

Cost estimate: 30 problems × K=16 × ~$0.002/call ≈ $1.

Usage:
    python -m alpha_program.exp_e8_aime --k 16 --output e8_aime2024.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.aime_tasks import AIMETask, load_aime_tasks
from alpha_program.aime_verifier import run_response


SYSTEM_PROMPT = """You are an expert mathematician solving AIME problems.
The American Invitational Mathematics Examination (AIME) asks for an integer
answer in [0, 999].

Read the problem carefully, work through the math step by step inside your
head if needed, then write a single Python function `def solve()` that
COMPUTES the answer numerically and returns it as a Python int.

You may import: math, fractions, decimal, itertools, functools, collections,
sympy, numpy, statistics, random, operator, bisect, heapq, re. Sympy is
encouraged for symbolic algebra. NO file I/O, NO networking, NO os/sys.

Output your code in a single ```python fenced block with `def solve():`
returning the integer. The function will be executed and the integer it
returns will be graded against the ground truth. Do NOT just print the
answer — RETURN it.
"""


def format_problem_prompt(problem: str) -> str:
    return (
        "Problem:\n"
        f"{problem}\n\n"
        "Write a `def solve()` function that returns the integer answer."
    )


# ════════════════════════════════════════════════════════════════════════
# Per-sample
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    sample_idx: int
    parse_ok: bool
    run_ok: bool
    answer: int | None
    error: str | None
    cost_usd: float
    tokens_in: int
    tokens_out: int


def run_one_sample(client: LLMClient, task: AIMETask, sample_idx: int,
                   temperature: float = 0.9, max_tokens: int = 4096,
                   timeout_s: float = 10.0) -> SampleResult:
    user = format_problem_prompt(task.problem)
    client.seed = 42 + sample_idx  # diverse seeds
    try:
        resp = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return SampleResult(
            sample_idx=sample_idx, parse_ok=False, run_ok=False,
            answer=None, error=f"API: {type(e).__name__}: {e}",
            cost_usd=0.0, tokens_in=0, tokens_out=0,
        )
    r = run_response(resp.content, timeout_s=timeout_s)
    return SampleResult(
        sample_idx=sample_idx,
        parse_ok=("def solve" in resp.content),
        run_ok=r.ok,
        answer=r.answer,
        error=r.error,
        cost_usd=resp.cost_usd,
        tokens_in=resp.input_tokens,
        tokens_out=resp.output_tokens,
    )


@dataclass
class TaskResult:
    task_id: str
    answer: int  # ground truth
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def n_run_ok(self) -> int:
        return sum(1 for s in self.samples if s.run_ok)

    @property
    def majority_answer(self) -> int | None:
        """Majority vote among run_ok samples; None if no valid answer."""
        valid = [s.answer for s in self.samples if s.run_ok and s.answer is not None]
        if not valid:
            return None
        ctr = Counter(valid)
        return ctr.most_common(1)[0][0]

    @property
    def first_valid_answer(self) -> int | None:
        """First valid answer (the deployable single-shot baseline)."""
        for s in self.samples:
            if s.run_ok and s.answer is not None:
                return s.answer
        return None

    @property
    def majority_correct(self) -> bool:
        return self.majority_answer == self.answer

    @property
    def first_valid_correct(self) -> bool:
        return self.first_valid_answer == self.answer

    @property
    def any_correct(self) -> bool:
        return any(s.run_ok and s.answer == self.answer for s in self.samples)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--n", type=int, default=-1, help="number of problems (-1 = all 30)")
    p.add_argument("--k", type=int, default=16, help="samples per problem")
    p.add_argument("--model", default="qwen-22b")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("=" * 70)
    print(f"E8: AlphaProgram on AIME {args.year}")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  K samples   : {args.k}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Workers     : {args.workers}")
    print()

    tasks = load_aime_tasks(year=args.year, n=args.n, seed=args.seed)
    print(f"Loaded {len(tasks)} AIME-{args.year} problems")
    client = LLMClient(args.model, seed=args.seed)
    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    results: list[TaskResult] = [
        TaskResult(task_id=t.id, answer=t.answer) for t in tasks
    ]
    progress_lock = Lock()
    save_lock = Lock()
    last_save = [0.0]
    SAVE_INTERVAL = 30
    n_tasks_complete = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {}
        for ti, task in enumerate(tasks):
            for si in range(args.k):
                f = pool.submit(
                    run_one_sample, client, task, si,
                    args.temperature, args.max_tokens, args.timeout_s,
                )
                future_to_idx[f] = (ti, si)
        for f in as_completed(future_to_idx):
            ti, si = future_to_idx[f]
            try:
                sr = f.result()
            except Exception as e:
                sr = SampleResult(sample_idx=si, parse_ok=False, run_ok=False,
                                  answer=None, error=f"runner: {e}",
                                  cost_usd=0.0, tokens_in=0, tokens_out=0)
            with progress_lock:
                results[ti].samples.append(sr)
                if len(results[ti].samples) == args.k:
                    results[ti].samples.sort(key=lambda s: s.sample_idx)
                    n_tasks_complete += 1
                    sc = "✓" if results[ti].majority_correct else "·"
                    print(f"[{n_tasks_complete}/{len(tasks)}] {tasks[ti].id} "
                          f"{sc} maj={results[ti].majority_answer} (gt={results[ti].answer}) "
                          f"any_correct={results[ti].any_correct} "
                          f"run_ok={results[ti].n_run_ok}/{args.k}",
                          flush=True)
                now = time.time()
                if now - last_save[0] > SAVE_INTERVAL:
                    last_save[0] = now
                    with save_lock:
                        save(results, output_path, args, t0)

    save(results, output_path, args, t0)
    elapsed = time.time() - t0

    n_majority_correct = sum(1 for r in results if r.majority_correct)
    n_first_correct = sum(1 for r in results if r.first_valid_correct)
    n_oracle = sum(1 for r in results if r.any_correct)
    n = len(results)

    maj_lo, maj_hi = wilson_ci(n_majority_correct, n)
    total_cost = sum(s.cost_usd for r in results for s in r.samples)
    cost_per_task = total_cost / n if n else 0
    cpc = (cost_per_task / (n_majority_correct / n)) if n_majority_correct > 0 else float("inf")

    print()
    print("=" * 70)
    print("FINAL — E8 AIME 2024")
    print("=" * 70)
    print(f"  Majority vote (deployable):  {n_majority_correct}/{n} = {n_majority_correct/n:.1%}  Wilson CI [{maj_lo:.1%}, {maj_hi:.1%}]")
    print(f"  First valid (single-shot):   {n_first_correct}/{n} = {n_first_correct/n:.1%}")
    print(f"  Oracle (any-correct):        {n_oracle}/{n} = {n_oracle/n:.1%}")
    print(f"  Cost: ${total_cost:.4f} total, ${cost_per_task:.4f}/task, ${cpc:.3f}/correct")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()
    print("Published frontier on AIME 2024 (cost-per-correct comparison):")
    refs = [
        ("GPT-4o", 0.09, 0.05),
        ("Claude 3.5 Sonnet", 0.16, 0.10),
        ("DeepSeek-V3", 0.40, 0.005),
        ("o1-preview", 0.52, 0.50),
        ("o1", 0.83, 1.50),
        ("DeepSeek-R1", 0.80, 0.30),
    ]
    print(f"  {'model':30s} {'rate':>7s} {'$/task':>10s} {'$/correct':>12s}")
    for label, rate, ct in refs:
        cpc_ref = ct / rate if rate > 0 else float("inf")
        print(f"  {label:30s} {rate:>6.0%}  ${ct:>7.4f}  ${cpc_ref:>9.4f}")
    print(f"  {'AlphaProgram (this work)':30s} {n_majority_correct/n:>6.0%}  ${cost_per_task:>7.4f}  ${cpc:>9.4f}")


def save(results: list[TaskResult], output_path: Path, args, t0: float) -> None:
    n_majority_correct = sum(1 for r in results if r.majority_correct and len(r.samples) >= args.k)
    n_first_correct = sum(1 for r in results if r.first_valid_correct and len(r.samples) >= args.k)
    n_oracle = sum(1 for r in results if r.any_correct and len(r.samples) >= args.k)
    n = sum(1 for r in results if len(r.samples) >= args.k)
    total_cost = sum(s.cost_usd for r in results for s in r.samples)
    out = {
        "experiment": "e8_aime",
        "year": args.year,
        "model": args.model,
        "n_planned": args.n if args.n > 0 else 30,
        "n_completed": n,
        "k_samples": args.k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "elapsed_s": time.time() - t0,
        "summary": {
            "majority_vote_correct": n_majority_correct,
            "majority_vote_rate": n_majority_correct / n if n else 0,
            "first_valid_correct": n_first_correct,
            "first_valid_rate": n_first_correct / n if n else 0,
            "oracle_correct": n_oracle,
            "oracle_rate": n_oracle / n if n else 0,
            "total_cost_usd": total_cost,
            "cost_per_task": total_cost / n if n else 0,
        },
        "per_task": [
            {
                "task_id": r.task_id,
                "ground_truth": r.answer,
                "n_samples": len(r.samples),
                "n_run_ok": r.n_run_ok,
                "majority_answer": r.majority_answer,
                "majority_correct": r.majority_correct,
                "first_valid_answer": r.first_valid_answer,
                "first_valid_correct": r.first_valid_correct,
                "any_correct": r.any_correct,
                "samples": [
                    {
                        "i": s.sample_idx,
                        "run_ok": s.run_ok,
                        "answer": s.answer,
                        "error": s.error[:120] if s.error else None,
                        "cost_usd": s.cost_usd,
                        "tokens_in": s.tokens_in,
                        "tokens_out": s.tokens_out,
                    }
                    for s in r.samples
                ],
            }
            for r in results
        ],
    }
    json.dump(out, open(output_path, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
