"""Unified validation runner — parameterized by model, dataset, n_tasks, K.

Used by V1 (AlphaProgram on the official 400-task ARC-AGI-1 public eval),
V3 (AlphaProgram on a 100-task ARC-AGI-2 pilot), V4 (qwen-coder-32b A/B
test on the 30-task battery), and V5 (K-sweep).

Usage:
    python -m alpha_program.run_validation \
        --model qwen-coder-32b --dataset arc1 --split training \
        --n 30 --k 32 --output v4_qwen_coder_32b.json

    python -m alpha_program.run_validation \
        --model qwen-22b --dataset arc1 --split evaluation \
        --n 400 --k 32 --output v1_arc1_eval_qwen22b.json

    python -m alpha_program.run_validation \
        --model qwen-22b --dataset arc2 --split test \
        --n 100 --k 32 --output v3_arc2_pilot.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, load_arc2_tasks
from alpha_program.exp_e2_verifier_loop import (
    run_one_sample,
    summarize,
    TaskResult,
    CandidateResult,
    SYSTEM_PROMPT,
)
from alpha_program import exp_e2_verifier_loop as base


def load_dataset_split(dataset: str, split: str, n: int, seed: int):
    if dataset == "arc1":
        return load_arc_tasks(n=n, seed=seed, split=split)
    if dataset == "arc2":
        return load_arc2_tasks(n=n, seed=seed, split=split)
    raise ValueError(f"unknown dataset {dataset!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model key in src/config.py MODELS")
    p.add_argument("--dataset", choices=["arc1", "arc2"], default="arc1")
    p.add_argument("--split", default="training",
                   help="arc1: training|evaluation|trial; arc2: train|test")
    p.add_argument("--n", type=int, default=30, help="number of tasks (-1 = full split)")
    p.add_argument("--k", type=int, default=32, help="programs per task")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="output JSON filename in results/alpha_program/")
    p.add_argument("--label", default="", help="human-readable label for the run")
    args = p.parse_args()

    # Push knobs into the base module so run_one_task picks them up
    base.K_PROGRAMS = args.k
    base.TEMPERATURE = args.temperature
    base.MAX_TOKENS = args.max_tokens
    base.WORKERS = args.workers
    base.SEED_BASE = args.seed
    base.MODEL_KEY = args.model

    print("=" * 70)
    print(f"Validation run: {args.label or args.output}")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.dataset} / {args.split}")
    print(f"  N tasks     : {args.n}")
    print(f"  K programs  : {args.k}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Workers     : {args.workers}")
    print(f"  Seed        : {args.seed}")
    print()

    tasks = load_dataset_split(args.dataset, args.split, args.n, args.seed)
    print(f"Loaded {len(tasks)} tasks")

    client = LLMClient(args.model, seed=args.seed)

    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    # ──────────────────────────────────────────────────────────────────────
    # Cross-task parallelism: every (task_index, sample_index) pair becomes
    # one job, and we run them all through a single thread pool. This makes
    # the total in-flight concurrency = args.workers, regardless of how K
    # programs are distributed across N tasks. Per-task results are
    # assembled as samples land.
    # ──────────────────────────────────────────────────────────────────────
    results: list[TaskResult] = [
        TaskResult(task_id=t.id, n_train_pairs=len(t.train_pairs))
        for t in tasks
    ]
    n_jobs_total = len(tasks) * args.k
    n_done = 0
    n_tasks_complete = 0
    progress_lock = Lock()
    save_lock = Lock()
    last_save_at = [0]  # mutable closure
    SAVE_INTERVAL_S = 30  # throttle incremental saves

    t0 = time.time()

    def submit_jobs(pool):
        for ti, task in enumerate(tasks):
            for si in range(args.k):
                yield pool.submit(run_one_sample, client, task, si), (ti, si)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {}
        for ti, task in enumerate(tasks):
            for si in range(args.k):
                f = pool.submit(run_one_sample, client, task, si)
                future_to_idx[f] = (ti, si)
        for f in as_completed(future_to_idx):
            ti, si = future_to_idx[f]
            try:
                cr = f.result()
            except Exception as e:
                cr = CandidateResult(
                    sample_idx=si,
                    response_text=f"<RUNNER_ERROR: {e}>",
                    parse_error=True,
                    train_passed=0,
                    train_total=len(tasks[ti].train_pairs),
                    train_score=0.0,
                    test_correct=None,
                    cost_usd=0.0,
                    tokens_in=0,
                    tokens_out=0,
                )
            with progress_lock:
                results[ti].candidates.append(cr)
                n_done += 1
                # task is complete when it has K candidates
                task_complete = len(results[ti].candidates) == args.k
                if task_complete:
                    results[ti].candidates.sort(key=lambda c: c.sample_idx)
                    n_tasks_complete += 1
                    elapsed = time.time() - t0
                    rate_so_far = sum(1 for r in results if r.candidates and r.best_program_test_correct) / max(n_tasks_complete, 1)
                    deploy_so_far = sum(1 for r in results if r.candidates and r.best_program_test_correct)
                    print(
                        f"[{n_tasks_complete}/{len(tasks)}] {tasks[ti].id} "
                        f"first_passing={results[ti].best_program_test_correct} "
                        f"running_rate={deploy_so_far}/{n_tasks_complete}={rate_so_far:.1%} "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                # throttled incremental save
                now = time.time()
                if now - last_save_at[0] > SAVE_INTERVAL_S or task_complete and n_tasks_complete == len(tasks):
                    last_save_at[0] = now
                    with save_lock:
                        save(results, output_path, args, t0)

    elapsed = time.time() - t0
    summary = summarize(results)

    print("\n" + "=" * 70)
    print(f"FINAL — {args.label or args.output}")
    print("=" * 70)
    print(f"  per-attempt rate          : {summary['per_attempt_rate']:7.1%}")
    print(f"  deployable (first-passing): {summary['deployable_k']:>3d}/{summary['n_tasks']} = {summary['deployable_rate']:7.1%}")
    print(f"  oracle (any-correct)      : {summary['oracle_k']:>3d}/{summary['n_tasks']} = {summary['oracle_rate']:7.1%}")
    print(f"  any-train-solved          : {summary['any_train_solved_k']:>3d}/{summary['n_tasks']} = {summary['any_train_solved_rate']:7.1%}")
    print(f"  parse error rate          : {summary['parse_error_rate']:7.1%}")
    print(f"  total cost                : ${summary['total_cost_usd']:.4f}")
    print(f"  tokens (in / out)         : {summary['total_tokens_in']:,} / {summary['total_tokens_out']:,}")
    print(f"  elapsed                   : {elapsed:.0f}s")


def save(results: list[TaskResult], output_path: Path, args, t0: float) -> None:
    n_complete = sum(1 for r in results if len(r.candidates) >= args.k)
    out = {
        "experiment": "validation_run",
        "label": args.label or args.output,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_tasks_planned": args.n,
        "n_tasks_completed": n_complete,
        "k_programs": args.k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "elapsed_s": time.time() - t0,
        "summary": summarize(results),
        "per_task": [
            {
                "task_id": r.task_id,
                "n_train_pairs": r.n_train_pairs,
                "per_attempt_rate": r.per_attempt_rate,
                "best_train_score": r.best_train_score,
                "any_train_solved": r.any_train_solved,
                "first_passing_test_correct": r.best_program_test_correct,
                "parse_error_rate": r.parse_error_rate,
                "candidates": [
                    {
                        "i": c.sample_idx,
                        "train_passed": c.train_passed,
                        "train_total": c.train_total,
                        "train_score": c.train_score,
                        "test_correct": c.test_correct,
                        "parse_error": c.parse_error,
                        "cost_usd": c.cost_usd,
                        "tokens_in": c.tokens_in,
                        "tokens_out": c.tokens_out,
                    }
                    for c in r.candidates
                ],
            }
            for r in results
        ],
    }
    json.dump(out, open(output_path, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
