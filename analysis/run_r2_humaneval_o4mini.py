"""R2: Fair HumanEval comparison — o4-mini at K=16 with execution filtering.

Same architecture, same K, same execution filter as our qwen-22b run.
This gives a same-metric, same-task comparison.

Uses OpenAI API directly (not Together AI).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
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
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI
from datasets import load_dataset


SYSTEM_PROMPT = """You are an expert Python programmer solving HumanEval problems.
You will be given a function signature with a docstring. Write the complete
function implementation. Output ONLY the function in a ```python fenced block.
Do NOT include test code. Do NOT add imports unless absolutely necessary.
"""

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_solution(text, entry_point):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for block in reversed(_FENCE_RE.findall(text)):
        if f"def {entry_point}" in block:
            return block.strip()
    if f"def {entry_point}" in text:
        idx = text.find(f"def {entry_point}")
        c = text[idx:]
        if "```" in c:
            c = c.split("```", 1)[0]
        return c.strip()
    return None


def _run_tests(solution_code, test_code, entry_point, queue):
    try:
        full = solution_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        exec(compile(full, "<humaneval>", "exec"), {"__builtins__": __builtins__})
        queue.put(("ok", True))
    except Exception as e:
        queue.put(("fail", str(e)[:100]))


def run_solution(sol, test, entry_point, timeout=10):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_run_tests, args=(sol, test, entry_point, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(0.2)
        if p.is_alive():
            p.kill()
        return False
    if q.empty():
        return False
    tag, _ = q.get()
    return tag == "ok"


@dataclass
class TaskResult:
    task_id: str
    samples: list = field(default_factory=list)  # list of (passed: bool, cost: float)

    @property
    def pass_at_k(self):
        return any(s["passed"] for s in self.samples)

    @property
    def pass_at_1(self):
        return self.samples[0]["passed"] if self.samples else False


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-half), min(1, center+half))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--model", default="o4-mini")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"R2: {args.model} pass@{args.k} on HumanEval-164 with execution filtering")

    ds = load_dataset("openai/openai_humaneval", split="test")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = [TaskResult(task_id=ds[i]["task_id"]) for i in range(len(ds))]
    lock = Lock()
    t0 = time.time()
    n_tasks_done = 0

    def run_one(ti, si):
        item = ds[ti]
        user = f"Complete the following Python function:\n\n```python\n{item['prompt']}\n```"
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=4096,
                temperature=0.8,
            )
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            cost = ((usage.prompt_tokens * 1.10 + usage.completion_tokens * 4.40) / 1e6) if usage else 0
        except Exception as e:
            return ti, si, {"passed": False, "cost": 0, "error": str(e)[:100]}

        sol = extract_solution(content, item["entry_point"])
        if sol is None:
            return ti, si, {"passed": False, "cost": cost, "error": "no function"}

        passed = run_solution(sol, item["test"], item["entry_point"])
        return ti, si, {"passed": passed, "cost": cost}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for ti in range(len(ds)):
            for si in range(args.k):
                f = pool.submit(run_one, ti, si)
                futures[f] = (ti, si)
        for f in as_completed(futures):
            ti, si, result = f.result()
            with lock:
                results[ti].samples.append(result)
                if len(results[ti].samples) == args.k:
                    n_tasks_done += 1
                    mark = "✓" if results[ti].pass_at_k else "·"
                    n_pass = sum(1 for r in results if len(r.samples) >= args.k and r.pass_at_k)
                    print(f"  [{n_tasks_done}/{len(ds)}] {results[ti].task_id} {mark} "
                          f"running pass@K={n_pass}/{n_tasks_done}", flush=True)

    elapsed = time.time() - t0
    n = len(results)
    pk = sum(1 for r in results if r.pass_at_k)
    p1 = sum(1 for r in results if r.pass_at_1)
    total_cost = sum(s["cost"] for r in results for s in r.samples)
    lok, hik = wilson(pk, n)
    lo1, hi1 = wilson(p1, n)

    print(f"\n{'='*60}")
    print(f"R2 FINAL: {args.model} on HumanEval-164, K={args.k}")
    print(f"{'='*60}")
    print(f"  pass@1:  {p1}/{n} = {p1/n:.1%}  [{lo1:.1%}, {hi1:.1%}]")
    print(f"  pass@K:  {pk}/{n} = {pk/n:.1%}  [{lok:.1%}, {hik:.1%}]")
    print(f"  Cost: ${total_cost:.2f} total, ${total_cost/n:.4f}/task")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"\n  Our qwen-22b:  pass@16 = 159/164 = 97.0%  at $0.003/task")
    print(f"  {args.model}:     pass@{args.k} = {pk}/{n} = {pk/n:.1%}  at ${total_cost/n:.4f}/task")

    out = {
        "experiment": "r2_humaneval_fair",
        "model": args.model,
        "k": args.k,
        "n": n,
        "pass_at_1": p1, "pass_at_1_rate": p1/n,
        "pass_at_k": pk, "pass_at_k_rate": pk/n,
        "total_cost": total_cost,
        "elapsed_s": elapsed,
        "per_task": [
            {"task_id": r.task_id, "pass_at_k": r.pass_at_k, "pass_at_1": r.pass_at_1,
             "n_passed": sum(1 for s in r.samples if s["passed"]),
             "cost": sum(s["cost"] for s in r.samples)}
            for r in results
        ],
    }
    outpath = Path("results/alpha_program") / args.output
    json.dump(out, open(outpath, "w"), indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
