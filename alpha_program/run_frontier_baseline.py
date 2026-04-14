"""V2 — Frontier-reasoner baseline on the same task set as V1.

Runs a single-shot direct-grid-output baseline (no AlphaProgram architecture)
on a frontier reasoner like DeepSeek-R1, so we can compare apples-to-apples
on the cost-correctness Pareto for the headline figure.

The reasoner is asked to output the test grid directly in the same format
the existing pivot-a ARC tasks use (rows of space-separated integers,
delimited by OUTPUT:/END_OUTPUT). We use ARCTask.check() to grade.

Usage:
    python -m alpha_program.run_frontier_baseline \
        --model deepseek-r1 --dataset arc1 --split evaluation \
        --n 400 --output v2_arc1_eval_deepseek_r1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import ARCTask, load_arc_tasks, load_arc2_tasks


SYSTEM_PROMPT = """You are an expert at solving ARC-AGI visual reasoning puzzles.

You will be given several input/output grid pairs that demonstrate a transformation
rule. Your job is to figure out the rule and apply it to a new input grid.

Output ONLY the resulting grid, in the same format as the examples (rows of
space-separated integers). Wrap your final answer in OUTPUT: ... END_OUTPUT
markers. Do not output anything else after END_OUTPUT.
"""


@dataclass
class FrontierResult:
    task_id: str
    correct: bool
    cost_usd: float
    tokens_in: int
    tokens_out: int
    response_text: str
    error: str | None = None


def run_one_task(client: LLMClient, task: ARCTask, max_tokens: int = 8192) -> FrontierResult:
    user = task.format_prompt()
    try:
        resp = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user,
            temperature=0.0,
            max_tokens=max_tokens,
            thinking=True,  # let DeepSeek-R1 / QwQ think
        )
    except Exception as e:
        return FrontierResult(
            task_id=task.id,
            correct=False,
            cost_usd=0.0,
            tokens_in=0,
            tokens_out=0,
            response_text="",
            error=f"{type(e).__name__}: {e}",
        )
    correct = task.check(resp.content)
    return FrontierResult(
        task_id=task.id,
        correct=correct,
        cost_usd=resp.cost_usd,
        tokens_in=resp.input_tokens,
        tokens_out=resp.output_tokens,
        response_text=resp.content,
    )


def load_dataset_split(dataset: str, split: str, n: int, seed: int):
    if dataset == "arc1":
        return load_arc_tasks(n=n, seed=seed, split=split)
    if dataset == "arc2":
        return load_arc2_tasks(n=n, seed=seed, split=split)
    raise ValueError(f"unknown dataset {dataset!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["arc1", "arc2"], default="arc1")
    p.add_argument("--split", default="evaluation")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    print("=" * 70)
    print(f"Frontier baseline: {args.label or args.output}")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.dataset} / {args.split}")
    print(f"  N tasks     : {args.n}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Workers     : {args.workers}")
    print()

    tasks = load_dataset_split(args.dataset, args.split, args.n, args.seed)
    print(f"Loaded {len(tasks)} tasks")

    client = LLMClient(args.model, seed=args.seed)
    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    results: list[FrontierResult] = []
    t0 = time.time()

    # Parallelize across tasks
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_task, client, t, args.max_tokens): t for t in tasks}
        n_done = 0
        for f in as_completed(futures):
            t = futures[f]
            try:
                r = f.result()
            except Exception as e:
                r = FrontierResult(
                    task_id=t.id, correct=False, cost_usd=0.0,
                    tokens_in=0, tokens_out=0, response_text="",
                    error=f"runner: {e}"
                )
            results.append(r)
            n_done += 1
            n_correct = sum(1 for x in results if x.correct)
            cost = sum(x.cost_usd for x in results)
            print(f"  [{n_done:>3}/{len(tasks)}] {r.task_id}: correct={r.correct} "
                  f"running={n_correct}/{n_done} ({n_correct/n_done:.1%}) "
                  f"cost=${cost:.3f}", flush=True)
            # incremental save
            save(results, output_path, args, t0)

    elapsed = time.time() - t0
    n_correct = sum(1 for r in results if r.correct)
    n_total = len(results)
    rate = n_correct / n_total if n_total else 0
    total_cost = sum(r.cost_usd for r in results)
    total_in = sum(r.tokens_in for r in results)
    total_out = sum(r.tokens_out for r in results)

    # Wilson 95% CI
    from math import sqrt
    z = 1.96
    p = rate
    n = n_total
    if n > 0:
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        ci_lo, ci_hi = max(0, center - half), min(1, center + half)
    else:
        ci_lo = ci_hi = 0

    print("\n" + "=" * 70)
    print(f"FINAL — {args.label or args.output}")
    print("=" * 70)
    print(f"  correctness  : {n_correct}/{n_total} = {rate:.1%}  Wilson 95% CI [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"  total cost   : ${total_cost:.4f}  (${total_cost/n_total if n_total else 0:.4f}/task)")
    print(f"  tokens (in/out): {total_in:,} / {total_out:,}")
    print(f"  elapsed      : {elapsed:.0f}s")


def save(results: list[FrontierResult], output_path: Path, args, t0: float) -> None:
    n_correct = sum(1 for r in results if r.correct)
    out = {
        "experiment": "frontier_baseline",
        "label": args.label or args.output,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_tasks_planned": args.n,
        "n_tasks_completed": len(results),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "elapsed_s": time.time() - t0,
        "n_correct": n_correct,
        "rate": n_correct / len(results) if results else 0.0,
        "total_cost_usd": sum(r.cost_usd for r in results),
        "total_tokens_in": sum(r.tokens_in for r in results),
        "total_tokens_out": sum(r.tokens_out for r in results),
        "per_task": [
            {
                "task_id": r.task_id,
                "correct": r.correct,
                "cost_usd": r.cost_usd,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "error": r.error,
                # truncate response to keep file size manageable
                "response_preview": r.response_text[-500:] if r.response_text else "",
            }
            for r in results
        ],
    }
    json.dump(out, open(output_path, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
