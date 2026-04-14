"""E3 — Verifier-in-loop best-of-N + critic loop on near-misses.

Builds on E2-bis: instead of running K=32 cold samples and stopping, we run
K=16 initial samples, then for each task identify the top partial-match
programs (highest train_score < 1.0) and feed them back to qwen-22b along
with the SPECIFIC failing pair as structured "this gave X but expected Y"
context. Two refinement rounds. Final pool = round-1 + round-2 + round-3.

Hypothesis: critic loop adds +5pp on the 30-task battery by recovering the
4 oracle-only tasks (where the right test answer exists in the K=32 pool
but no program is verified) plus the 1 verified-but-wrong task (arc_0101).
Conservative target: 43% → 48%.

Run after V1 lands so we know the floor we're building from.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, load_arc2_tasks
from alpha_program.dsl import DSL_DOC
from alpha_program.exp_e2_verifier_loop import (
    SYSTEM_PROMPT, format_task_prompt, run_one_sample,
    TaskResult, CandidateResult, summarize,
)
from alpha_program.verifier import score_response, score_program, _grids_equal


# ════════════════════════════════════════════════════════════════════════
# Critic prompt
# ════════════════════════════════════════════════════════════════════════

CRITIC_SYSTEM = (
    SYSTEM_PROMPT
    + "\n\nYou will be given a previous attempt at writing the `transform(grid)` "
    "function that PARTIALLY succeeded — it produced the correct output for "
    "some training examples but the wrong output for others. Your job is to "
    "WRITE A NEW VERSION of the function that correctly handles the failing "
    "case as well, while still preserving the cases that already worked.\n\n"
    "Read the failing example carefully — note exactly which cells are wrong "
    "and how. Then revise the logic. Output only the corrected `def transform(grid):` "
    "in a ```python fenced block, no preamble."
)


def format_critic_prompt(
    task_train_pairs: list[tuple[list[list[int]], list[list[int]]]],
    test_input: list[list[int]],
    program_src: str,
    failing_pairs: list[tuple[int, list[list[int]], list[list[int]]]],
    passing_pair_idxs: list[int],
) -> str:
    """Build the critic's user prompt with structured failure feedback."""
    lines = [
        "Here are the training input/output pairs for one ARC puzzle:\n",
    ]
    for i, (inp, out) in enumerate(task_train_pairs):
        lines.append(f"# Example {i+1}")
        lines.append(f"input  = {inp!r}")
        lines.append(f"output = {out!r}")
        lines.append("")
    lines.append(f"# The held-out test input:")
    lines.append(f"test_input = {test_input!r}")
    lines.append("")
    lines.append("Your previous attempt was:")
    lines.append("```python")
    lines.append(program_src.strip())
    lines.append("```")
    lines.append("")
    if passing_pair_idxs:
        lines.append(f"This worked for examples: {sorted([i+1 for i in passing_pair_idxs])}")
    lines.append(f"It FAILED for examples: {sorted([i+1 for i, _, _ in failing_pairs])}")
    lines.append("")
    for pair_idx, expected, got in failing_pairs[:2]:  # show at most 2 failing pairs to keep prompt small
        lines.append(f"For example {pair_idx+1}:")
        lines.append(f"  expected = {expected!r}")
        lines.append(f"  your result = {got!r}")
        lines.append("")
    lines.append("Write a CORRECTED `def transform(grid):` that handles all of the training examples correctly. Output only the function in a ```python fenced block.")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Critic candidate sampler
# ════════════════════════════════════════════════════════════════════════

def run_critic_sample(
    client: LLMClient,
    task,
    base_candidate: CandidateResult,
    sample_idx: int,
    timeout_s: float = 2.0,
) -> CandidateResult:
    """Generate one critic-driven candidate by feeding back the failure of `base_candidate`."""
    from alpha_program.verifier import extract_program, run_program
    base_src = extract_program(base_candidate.response_text)
    if base_src is None:
        return CandidateResult(
            sample_idx=sample_idx,
            response_text="<no_base_program>",
            parse_error=True,
            train_passed=0,
            train_total=len(task.train_pairs),
            train_score=0.0,
            test_correct=None,
            cost_usd=0.0,
            tokens_in=0,
            tokens_out=0,
        )

    # rebuild the failing-pair details by re-running base on each train pair
    failing = []
    passing = []
    for pi, (inp, expected) in enumerate(task.train_pairs):
        r = run_program(base_src, inp, timeout_s=timeout_s)
        if r.ok and r.output is not None and len(r.output) == len(expected) and all(
            len(a) == len(b) and all(x == y for x, y in zip(a, b))
            for a, b in zip(r.output, expected)
        ):
            passing.append(pi)
        else:
            got = r.output if r.ok else [["error"]]
            failing.append((pi, expected, got))

    if not failing:
        # already passes everything — no critic needed
        return base_candidate

    user = format_critic_prompt(task.train_pairs, task.test_input, base_src, failing, passing)
    client.seed = sample_idx + 1000  # different seed space from initial samples
    try:
        resp = client.generate(
            system_prompt=CRITIC_SYSTEM,
            user_prompt=user,
            temperature=0.7,  # lower than initial sampling — we want focused fixes, not wild diversity
            max_tokens=1024,
        )
    except Exception as e:
        return CandidateResult(
            sample_idx=sample_idx,
            response_text=f"<API_ERROR: {type(e).__name__}: {e}>",
            parse_error=True,
            train_passed=0,
            train_total=len(task.train_pairs),
            train_score=0.0,
            test_correct=None,
            cost_usd=0.0,
            tokens_in=0,
            tokens_out=0,
        )

    sr = score_response(
        resp.content,
        task.train_pairs,
        test_input=task.test_input,
        timeout_s=timeout_s,
    )
    test_correct = None
    if sr.test_output is not None:
        test_correct = _grids_equal(sr.test_output, task.test_output)

    return CandidateResult(
        sample_idx=sample_idx,
        response_text=resp.content,
        parse_error=sr.parse_error,
        train_passed=sr.n_passed,
        train_total=sr.n_train,
        train_score=sr.score,
        test_correct=test_correct,
        cost_usd=resp.cost_usd,
        tokens_in=resp.input_tokens,
        tokens_out=resp.output_tokens,
    )


# ════════════════════════════════════════════════════════════════════════
# Main loop: K_INITIAL cold samples + N_REFINE rounds of M_CRITIC critic samples
# ════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen-22b")
    p.add_argument("--dataset", choices=["arc1", "arc2"], default="arc1")
    p.add_argument("--split", default="training")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--k-initial", type=int, default=16, help="initial cold samples per task")
    p.add_argument("--n-refine-rounds", type=int, default=2)
    p.add_argument("--m-critic", type=int, default=8, help="critic samples per refinement round")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    print("=" * 70)
    print(f"E3 critic loop: {args.label or args.output}")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.dataset} / {args.split}")
    print(f"  N tasks     : {args.n}")
    print(f"  K initial   : {args.k_initial}")
    print(f"  Critic refine rounds: {args.n_refine_rounds}")
    print(f"  M critic per round  : {args.m_critic}")
    print(f"  Total candidates per task: {args.k_initial + args.n_refine_rounds * args.m_critic}")
    print()

    # Push the runtime knobs into the base module so run_one_sample picks up the right model
    from alpha_program import exp_e2_verifier_loop as base
    base.MODEL_KEY = args.model
    base.TEMPERATURE = args.temperature
    base.MAX_TOKENS = args.max_tokens
    base.SEED_BASE = args.seed
    base.K_PROGRAMS = args.k_initial
    base.WORKERS = args.workers

    if args.dataset == "arc1":
        tasks = load_arc_tasks(n=args.n, seed=args.seed, split=args.split)
    else:
        tasks = load_arc2_tasks(n=args.n, seed=args.seed, split=args.split)
    print(f"Loaded {len(tasks)} tasks")

    client = LLMClient(args.model, seed=args.seed)
    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    # ── Phase 1: K_INITIAL cold samples per task (parallel-across-everything)
    print(f"\n── Phase 1: cold sampling ({args.k_initial} per task, {args.k_initial * args.n} total)")
    results: list[TaskResult] = [
        TaskResult(task_id=t.id, n_train_pairs=len(t.train_pairs))
        for t in tasks
    ]
    progress_lock = Lock()
    save_lock = Lock()
    last_save = [0.0]
    SAVE_INTERVAL = 30
    t0 = time.time()
    n_tasks_complete_phase1 = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {}
        for ti, task in enumerate(tasks):
            for si in range(args.k_initial):
                f = pool.submit(run_one_sample, client, task, si)
                future_to_idx[f] = (ti, si)
        for f in as_completed(future_to_idx):
            ti, si = future_to_idx[f]
            try:
                cr = f.result()
            except Exception as e:
                cr = CandidateResult(sample_idx=si, response_text=f"<RUNNER_ERROR: {e}>",
                                     parse_error=True, train_passed=0, train_total=len(tasks[ti].train_pairs),
                                     train_score=0.0, test_correct=None, cost_usd=0.0, tokens_in=0, tokens_out=0)
            with progress_lock:
                results[ti].candidates.append(cr)
                if len(results[ti].candidates) == args.k_initial:
                    results[ti].candidates.sort(key=lambda c: c.sample_idx)
                    n_tasks_complete_phase1 += 1
                    if n_tasks_complete_phase1 % 5 == 0:
                        deploy = sum(1 for r in results if len(r.candidates) >= args.k_initial and r.best_program_test_correct)
                        print(f"  [phase1 {n_tasks_complete_phase1}/{len(tasks)}] deploy={deploy}, elapsed={time.time()-t0:.0f}s", flush=True)
                now = time.time()
                if now - last_save[0] > SAVE_INTERVAL:
                    last_save[0] = now
                    with save_lock:
                        save(results, output_path, args, t0, "phase1")

    print(f"  ✓ phase 1 complete in {time.time()-t0:.0f}s")
    save(results, output_path, args, t0, "phase1_done")

    # ── Phase 2: N_REFINE rounds of M_CRITIC samples per task on partial-pass programs
    for round_idx in range(args.n_refine_rounds):
        # for each task, find the best partial-pass programs (train_score in (0, 1))
        # and queue critic samples
        print(f"\n── Phase 2.{round_idx+1}: critic refinement on partial-pass programs")
        tasks_with_partial = []
        for ti, r in enumerate(results):
            if r.best_program_test_correct:
                continue  # already deployable; skip
            partial = sorted(
                [c for c in r.candidates if 0 < c.train_score < 1.0],
                key=lambda c: -c.train_score,
            )
            if not partial:
                continue
            tasks_with_partial.append((ti, partial[:args.m_critic]))
        print(f"  {len(tasks_with_partial)} tasks have partial-pass programs to refine")

        if not tasks_with_partial:
            break

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            f2idx = {}
            base_seed = args.k_initial + round_idx * args.m_critic
            for ti, candidates in tasks_with_partial:
                for j, base_cand in enumerate(candidates):
                    si = base_seed + j
                    f = pool.submit(run_critic_sample, client, tasks[ti], base_cand, si)
                    f2idx[f] = (ti, si)
            for f in as_completed(f2idx):
                ti, si = f2idx[f]
                try:
                    cr = f.result()
                except Exception as e:
                    cr = CandidateResult(sample_idx=si, response_text=f"<CRITIC_ERROR: {e}>",
                                         parse_error=True, train_passed=0, train_total=len(tasks[ti].train_pairs),
                                         train_score=0.0, test_correct=None, cost_usd=0.0, tokens_in=0, tokens_out=0)
                with progress_lock:
                    results[ti].candidates.append(cr)
                    now = time.time()
                    if now - last_save[0] > SAVE_INTERVAL:
                        last_save[0] = now
                        with save_lock:
                            save(results, output_path, args, t0, f"phase2.{round_idx+1}")

        save(results, output_path, args, t0, f"phase2.{round_idx+1}_done")

    # ── Final report
    elapsed = time.time() - t0
    summary = summarize(results)
    print("\n" + "=" * 70)
    print(f"FINAL — {args.label or args.output}")
    print("=" * 70)
    print(f"  per-attempt rate          : {summary['per_attempt_rate']:7.1%}")
    print(f"  deployable (first-passing): {summary['deployable_k']:>3d}/{summary['n_tasks']} = {summary['deployable_rate']:7.1%}")
    print(f"  oracle (any-correct)      : {summary['oracle_k']:>3d}/{summary['n_tasks']} = {summary['oracle_rate']:7.1%}")
    print(f"  any-train-solved          : {summary['any_train_solved_k']:>3d}/{summary['n_tasks']} = {summary['any_train_solved_rate']:7.1%}")
    print(f"  total cost                : ${summary['total_cost_usd']:.4f}")
    print(f"  elapsed                   : {elapsed:.0f}s")


def save(results, output_path, args, t0, phase="?"):
    out = {
        "experiment": "e3_critic_loop",
        "phase": phase,
        "label": args.label or args.output,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_tasks_planned": args.n,
        "k_initial": args.k_initial,
        "n_refine_rounds": args.n_refine_rounds,
        "m_critic": args.m_critic,
        "elapsed_s": time.time() - t0,
        "summary": summarize(results),
        "per_task": [
            {
                "task_id": r.task_id,
                "n_train_pairs": r.n_train_pairs,
                "n_candidates": len(r.candidates),
                "best_train_score": r.best_train_score,
                "any_train_solved": r.any_train_solved,
                "first_passing_test_correct": r.best_program_test_correct,
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
