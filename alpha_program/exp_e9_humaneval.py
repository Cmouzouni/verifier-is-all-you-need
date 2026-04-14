"""E9 — AlphaProgram on HumanEval with unit-test execution + self-consistency.

Each HumanEval problem provides a function signature + docstring, a set of
assert-based test cases, and an entry_point. The proposer writes the function
body; the verifier executes the tests. Selection is by majority vote among
solutions that PASS all public tests.

This is the third verification domain after ARC (grid execution) and AIME
(sympy execution), testing whether the proposer-verifier-vote template
generalizes to code.

Usage:
    python -m alpha_program.exp_e9_humaneval --k 16 --output e9_humaneval_k16.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR


# ════════════════════════════════════════════════════════════════════════
# Data loader
# ════════════════════════════════════════════════════════════════════════

@dataclass
class HumanEvalTask:
    task_id: str
    prompt: str           # function signature + docstring
    test: str             # assert-based test code
    entry_point: str      # function name
    canonical: str = ""   # reference solution (for oracle, not shown to model)


def load_humaneval(n: int = -1, seed: int = 42) -> list[HumanEvalTask]:
    from datasets import load_dataset
    import random
    ds = load_dataset("openai/openai_humaneval", split="test")
    if n < 0 or n >= len(ds):
        indices = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(ds)), n))
    return [
        HumanEvalTask(
            task_id=ds[i]["task_id"],
            prompt=ds[i]["prompt"],
            test=ds[i]["test"],
            entry_point=ds[i]["entry_point"],
            canonical=ds[i].get("canonical_solution", ""),
        )
        for i in indices
    ]


# ════════════════════════════════════════════════════════════════════════
# Verifier: execute solution + test cases in sandbox
# ════════════════════════════════════════════════════════════════════════

import re

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_solution(text: str, entry_point: str) -> str | None:
    """Extract the function body from an LLM response."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Prefer last fenced block containing the function
    blocks = _FENCE_RE.findall(text)
    for block in reversed(blocks):
        if f"def {entry_point}" in block:
            return block.strip()
    # Bare def in text
    if f"def {entry_point}" in text:
        idx = text.find(f"def {entry_point}")
        candidate = text[idx:]
        if "```" in candidate:
            candidate = candidate.split("```", 1)[0]
        return candidate.strip()
    return None


def _run_tests_subprocess(solution_code: str, test_code: str, entry_point: str, queue: mp.Queue):
    try:
        # Build full program: solution + tests
        full = solution_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        exec(compile(full, "<humaneval>", "exec"), {"__builtins__": __builtins__})
        queue.put(("ok", True))
    except AssertionError:
        queue.put(("fail", "assertion failed"))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {str(e)[:120]}"))


def run_solution(solution_code: str, test_code: str, entry_point: str, timeout_s: float = 10.0) -> tuple[bool, str | None]:
    """Execute solution against tests. Returns (passed, error_or_none)."""
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_run_tests_subprocess, args=(solution_code, test_code, entry_point, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(0.2)
        if p.is_alive():
            p.kill()
        return False, f"timeout after {timeout_s}s"
    if q.empty():
        return False, "subprocess died"
    tag, payload = q.get()
    if tag == "ok":
        return True, None
    return False, payload


# ════════════════════════════════════════════════════════════════════════
# System prompt
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert Python programmer solving HumanEval problems.
You will be given a function signature with a docstring. Write the complete
function implementation. Output ONLY the function in a ```python fenced block.
Do NOT include test code. Do NOT add imports unless absolutely necessary.
"""


# ════════════════════════════════════════════════════════════════════════
# Per-sample
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    sample_idx: int
    extracted: bool
    passed: bool
    error: str | None
    cost_usd: float
    tokens_in: int
    tokens_out: int
    solution_hash: str = ""  # hash of the solution for dedup voting


def run_one_sample(client: LLMClient, task: HumanEvalTask, sample_idx: int,
                   temperature: float = 0.8, max_tokens: int = 2048,
                   timeout_s: float = 10.0) -> SampleResult:
    user = f"Complete the following Python function:\n\n```python\n{task.prompt}\n```"
    client.seed = 42 + sample_idx
    try:
        resp = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return SampleResult(sample_idx=sample_idx, extracted=False, passed=False,
                           error=f"API: {e}", cost_usd=0, tokens_in=0, tokens_out=0)

    sol = extract_solution(resp.content, task.entry_point)
    if sol is None:
        return SampleResult(sample_idx=sample_idx, extracted=False, passed=False,
                           error="no function found", cost_usd=resp.cost_usd,
                           tokens_in=resp.input_tokens, tokens_out=resp.output_tokens)

    passed, err = run_solution(sol, task.test, task.entry_point, timeout_s=timeout_s)
    return SampleResult(
        sample_idx=sample_idx, extracted=True, passed=passed, error=err,
        cost_usd=resp.cost_usd, tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
        solution_hash=str(hash(sol)),
    )


@dataclass
class TaskResult:
    task_id: str
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for s in self.samples if s.passed)

    @property
    def pass_at_1(self) -> bool:
        return any(s.passed for s in self.samples[:1])

    @property
    def pass_at_k(self) -> bool:
        return any(s.passed for s in self.samples)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-half), min(1, center+half))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=-1, help="number of problems (-1 = all 164)")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--model", default="qwen-22b")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("=" * 70)
    print(f"E9: AlphaProgram on HumanEval")
    print("=" * 70)
    print(f"  Model: {args.model}, K={args.k}, n={args.n}, workers={args.workers}")
    print()

    tasks = load_humaneval(n=args.n, seed=args.seed)
    print(f"Loaded {len(tasks)} HumanEval problems")
    client = LLMClient(args.model, seed=args.seed)
    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    results = [TaskResult(task_id=t.task_id) for t in tasks]
    progress_lock = Lock()
    save_lock = Lock()
    last_save = [0.0]
    n_tasks_complete = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {}
        for ti, task in enumerate(tasks):
            for si in range(args.k):
                f = pool.submit(run_one_sample, client, task, si, args.temperature, args.max_tokens)
                future_to_idx[f] = (ti, si)
        for f in as_completed(future_to_idx):
            ti, si = future_to_idx[f]
            try:
                sr = f.result()
            except Exception as e:
                sr = SampleResult(sample_idx=si, extracted=False, passed=False,
                                 error=f"runner: {e}", cost_usd=0, tokens_in=0, tokens_out=0)
            with progress_lock:
                results[ti].samples.append(sr)
                if len(results[ti].samples) == args.k:
                    results[ti].samples.sort(key=lambda s: s.sample_idx)
                    n_tasks_complete += 1
                    n_pass_at_k = sum(1 for r in results if r.samples and r.pass_at_k)
                    sc = "✓" if results[ti].pass_at_k else "·"
                    print(f"[{n_tasks_complete}/{len(tasks)}] {tasks[ti].task_id} "
                          f"{sc} passed={results[ti].n_passed}/{args.k} "
                          f"running_pass@K={n_pass_at_k}/{n_tasks_complete}",
                          flush=True)
                now = time.time()
                if now - last_save[0] > 30:
                    last_save[0] = now
                    with save_lock:
                        save(results, output_path, args, t0)

    save(results, output_path, args, t0)
    elapsed = time.time() - t0

    n = len(results)
    n_pass1 = sum(1 for r in results if r.pass_at_1)
    n_passk = sum(1 for r in results if r.pass_at_k)
    total_cost = sum(s.cost_usd for r in results for s in r.samples)

    lo1, hi1 = wilson_ci(n_pass1, n)
    lok, hik = wilson_ci(n_passk, n)

    print()
    print("=" * 70)
    print("FINAL — E9 HumanEval")
    print("=" * 70)
    print(f"  pass@1:  {n_pass1}/{n} = {n_pass1/n:.1%}  [{lo1:.1%}, {hi1:.1%}]")
    print(f"  pass@K:  {n_passk}/{n} = {n_passk/n:.1%}  [{lok:.1%}, {hik:.1%}]")
    print(f"  Cost: ${total_cost:.4f} total, ${total_cost/n:.5f}/task")
    if n_passk > 0:
        cpc = (total_cost / n) / (n_passk / n)
        print(f"  Cost-per-correct (pass@K): ${cpc:.4f}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()
    print("Published frontier on HumanEval:")
    refs = [("GPT-4o", 0.91, 0.005), ("Claude 3.5 Sonnet", 0.92, 0.01),
            ("DeepSeek-V3", 0.88, 0.001), ("o1", 0.93, 0.15)]
    for label, rate, ct in refs:
        print(f"  {label:25s} pass@1={rate:.0%}  ${ct:.4f}/task  ${ct/rate:.4f}/correct")
    rate_ours = n_passk / n if n else 0
    ct_ours = total_cost / n if n else 0
    cpc_ours = ct_ours / rate_ours if rate_ours > 0 else float("inf")
    print(f"  {'AlphaProgram (this work)':25s} pass@K={rate_ours:.0%}  ${ct_ours:.5f}/task  ${cpc_ours:.4f}/correct")


def save(results, output_path, args, t0):
    n = sum(1 for r in results if len(r.samples) >= args.k)
    n_pass1 = sum(1 for r in results if len(r.samples) >= args.k and r.pass_at_1)
    n_passk = sum(1 for r in results if len(r.samples) >= args.k and r.pass_at_k)
    total_cost = sum(s.cost_usd for r in results for s in r.samples)
    out = {
        "experiment": "e9_humaneval",
        "model": args.model,
        "n_planned": args.n if args.n > 0 else 164,
        "n_completed": n,
        "k_samples": args.k,
        "elapsed_s": time.time() - t0,
        "summary": {
            "pass_at_1": n_pass1, "pass_at_1_rate": n_pass1/n if n else 0,
            "pass_at_k": n_passk, "pass_at_k_rate": n_passk/n if n else 0,
            "total_cost_usd": total_cost,
            "cost_per_task": total_cost/n if n else 0,
        },
        "per_task": [
            {
                "task_id": r.task_id,
                "n_samples": len(r.samples),
                "n_passed": r.n_passed,
                "pass_at_1": r.pass_at_1 if len(r.samples) >= 1 else None,
                "pass_at_k": r.pass_at_k if len(r.samples) >= args.k else None,
                "samples": [
                    {"i": s.sample_idx, "extracted": s.extracted, "passed": s.passed,
                     "error": s.error[:80] if s.error else None,
                     "cost_usd": s.cost_usd, "tokens_in": s.tokens_in, "tokens_out": s.tokens_out}
                    for s in r.samples
                ],
            }
            for r in results
        ],
    }
    json.dump(out, open(output_path, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
