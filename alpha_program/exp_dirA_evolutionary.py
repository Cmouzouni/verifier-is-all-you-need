"""Direction A — Evolutionary refinement with inter-round information flow.

Unlike E3's simple critic ("fix this failing pair"), this is evolutionary:
the TOP-3 partial-pass programs from round N are shown to proposers in
round N+1 as context, with structural information about what works and
what doesn't. Information flows BETWEEN samples across rounds.

Design:
  Round 1: K=8 cold samples (standard — same as current architecture)
  Round 2: For tasks NOT yet solved in round 1, show the TOP-3 programs
           (by train_score) to K=8 NEW proposers as "previous attempts"
           with per-pair pass/fail diagnostics
  Round 3: Same, using best programs from rounds 1+2

Total budget per task: 8 + 8 + 8 = 24 candidates (same as K=24 cold)
Comparison: K=24 cold samples from the same model (fair budget match)

The hypothesis: inter-round information flow unlocks programs that cold
sampling doesn't, because later-round proposers can combine insights
from multiple partial solutions.

Usage:
    python -m alpha_program.exp_dirA_evolutionary \
        --n 30 --k-per-round 8 --rounds 3 \
        --output dirA_evolutionary_training30.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks
from alpha_program.dsl import DSL_DOC
from alpha_program.verifier import extract_program, score_program, run_program, _grids_equal
from alpha_program.exp_e2_verifier_loop import (
    SYSTEM_PROMPT as BASE_SYSTEM,
    format_task_prompt,
    run_one_sample,
    TaskResult, CandidateResult, summarize,
)
from alpha_program import exp_e2_verifier_loop as base


# ════════════════════════════════════════════════════════════════════════
# Evolutionary prompt: show previous attempts + per-pair diagnostics
# ════════════════════════════════════════════════════════════════════════

EVOLUTIONARY_SYSTEM = BASE_SYSTEM + """

IMPORTANT: You will be shown previous attempts at solving this puzzle.
Some passed certain training examples but failed others. Study them
carefully — understand what they got right and what they got wrong —
then write a NEW, IMPROVED version that handles ALL training examples.

Do NOT just copy a previous attempt. Combine the insights from multiple
attempts into a single better solution.
"""


def format_evolutionary_prompt(
    train_pairs: list[tuple[list, list]],
    test_input: list[list[int]],
    previous_attempts: list[dict],  # [{src, train_score, per_pair: [{i, passed, expected_shape, got_shape}]}]
) -> str:
    """Build the evolutionary prompt with previous attempts as context."""
    lines = ["Here are the training input/output pairs for one ARC puzzle:\n"]
    for i, (inp, out) in enumerate(train_pairs):
        lines.append(f"# Example {i+1}")
        lines.append(f"input  = {inp!r}")
        lines.append(f"output = {out!r}")
        lines.append("")
    lines.append(f"# The held-out test input:")
    lines.append(f"test_input = {test_input!r}")
    lines.append("")

    lines.append("=" * 60)
    lines.append(f"PREVIOUS ATTEMPTS ({len(previous_attempts)} shown, ranked by train-pair score):")
    lines.append("=" * 60)

    for rank, attempt in enumerate(previous_attempts, 1):
        score_pct = attempt["train_score"] * 100
        lines.append(f"\n--- Attempt #{rank} (passed {score_pct:.0f}% of training pairs) ---")
        lines.append("```python")
        lines.append(attempt["src"].strip())
        lines.append("```")

        # Show per-pair diagnostics
        for pp in attempt.get("per_pair", []):
            status = "✓ PASSED" if pp.get("passed") else "✗ FAILED"
            lines.append(f"  Example {pp['i']+1}: {status}")
            if not pp.get("passed"):
                if pp.get("error"):
                    lines.append(f"    Error: {pp['error'][:100]}")
                elif pp.get("expected_shape") and pp.get("got_shape"):
                    if pp["expected_shape"] != pp["got_shape"]:
                        lines.append(f"    Shape mismatch: expected {pp['expected_shape']}, got {pp['got_shape']}")
                    else:
                        lines.append(f"    Wrong values (shape matches: {pp['expected_shape']})")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Write a NEW `def transform(grid):` that passes ALL training examples.")
    lines.append("Combine insights from the attempts above. Output in a ```python block.")

    return "\n".join(lines)


def get_attempt_details(src: str, train_pairs: list[tuple[list, list]]) -> dict:
    """Run a program on all train pairs and return structured diagnostics."""
    from alpha_program.verifier import score_program
    sr = score_program(src, train_pairs, timeout_s=2.0)
    return {
        "src": src,
        "train_score": sr.score,
        "per_pair": sr.per_pair,
    }


def run_evolutionary_sample(
    client: LLMClient,
    task,
    sample_idx: int,
    previous_attempts: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 1024,
    timeout_s: float = 2.0,
) -> CandidateResult:
    """Generate one evolutionary candidate using previous attempts as context."""
    user = format_evolutionary_prompt(task.train_pairs, task.test_input, previous_attempts)
    client.seed = sample_idx + 2000  # different seed space
    try:
        resp = client.generate(
            system_prompt=EVOLUTIONARY_SYSTEM,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return CandidateResult(
            sample_idx=sample_idx, response_text=f"<API_ERROR: {e}>",
            parse_error=True, train_passed=0, train_total=len(task.train_pairs),
            train_score=0.0, test_correct=None, cost_usd=0.0, tokens_in=0, tokens_out=0,
        )

    from alpha_program.verifier import score_response
    sr = score_response(resp.content, task.train_pairs, test_input=task.test_input, timeout_s=timeout_s)
    test_correct = None
    if sr.test_output is not None:
        test_correct = _grids_equal(sr.test_output, task.test_output)

    return CandidateResult(
        sample_idx=sample_idx, response_text=resp.content,
        parse_error=sr.parse_error, train_passed=sr.n_passed, train_total=sr.n_train,
        train_score=sr.score, test_correct=test_correct,
        cost_usd=resp.cost_usd, tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
    )


# ════════════════════════════════════════════════════════════════════════
# Main loop: multi-round evolutionary refinement
# ════════════════════════════════════════════════════════════════════════

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-half), min(1, center+half))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen-22b")
    p.add_argument("--dataset", choices=["arc1", "arc2"], default="arc1")
    p.add_argument("--split", default="training")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--k-per-round", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--top-n-attempts", type=int, default=3, help="how many previous attempts to show")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Push config
    base.MODEL_KEY = args.model
    base.SEED_BASE = args.seed
    base.WORKERS = args.workers

    total_k = args.k_per_round * args.rounds
    print("=" * 70)
    print(f"Direction A: Evolutionary refinement ({args.rounds} rounds × K={args.k_per_round} = {total_k} total)")
    print("=" * 70)
    print(f"  Model: {args.model}, n={args.n}, top_attempts={args.top_n_attempts}")
    print()

    tasks = load_arc_tasks(n=args.n, seed=args.seed, split=args.split)
    print(f"Loaded {len(tasks)} tasks")
    client = LLMClient(args.model, seed=args.seed)
    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    # Per-task state: all candidates across all rounds
    results = [TaskResult(task_id=t.id, n_train_pairs=len(t.train_pairs)) for t in tasks]
    t0 = time.time()

    for round_idx in range(args.rounds):
        round_label = f"Round {round_idx+1}/{args.rounds}"
        is_cold = (round_idx == 0)

        if is_cold:
            print(f"\n── {round_label}: cold sampling (K={args.k_per_round}) ──")
        else:
            print(f"\n── {round_label}: evolutionary refinement (K={args.k_per_round}, top-{args.top_n_attempts} as context) ──")

        # Determine which tasks need more work
        tasks_to_run = []
        for ti, task in enumerate(tasks):
            if results[ti].best_program_test_correct:
                continue  # already solved
            tasks_to_run.append(ti)

        if not tasks_to_run:
            print(f"  All tasks solved! Stopping early.")
            break

        print(f"  {len(tasks_to_run)} tasks still unsolved")

        # For evolutionary rounds, prepare context for each unsolved task
        task_contexts: dict[int, list[dict]] = {}
        if not is_cold:
            for ti in tasks_to_run:
                # Get top-N candidates by train_score from all previous rounds
                all_cands = results[ti].candidates
                # Extract programs and score them for diagnostics
                attempts = []
                for c in all_cands:
                    if c.parse_error:
                        continue
                    src = extract_program(c.response_text)
                    if src is None:
                        continue
                    detail = get_attempt_details(src, tasks[ti].train_pairs)
                    if detail["train_score"] > 0:  # only show attempts that passed at least 1
                        attempts.append(detail)
                # Sort by train_score descending, take top N
                attempts.sort(key=lambda a: -a["train_score"])
                task_contexts[ti] = attempts[:args.top_n_attempts]

        # Submit jobs for this round
        n_done = 0
        progress_lock = Lock()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for ti in tasks_to_run:
                for si in range(args.k_per_round):
                    global_si = round_idx * args.k_per_round + si
                    if is_cold:
                        f = pool.submit(run_one_sample, client, tasks[ti], global_si)
                    else:
                        context = task_contexts.get(ti, [])
                        if not context:
                            # No partial-pass programs → fall back to cold sample
                            f = pool.submit(run_one_sample, client, tasks[ti], global_si)
                        else:
                            f = pool.submit(
                                run_evolutionary_sample, client, tasks[ti], global_si,
                                context, 0.8, 1024, 2.0,
                            )
                    futures[f] = (ti, global_si)

            for f in as_completed(futures):
                ti, si = futures[f]
                try:
                    cr = f.result()
                except Exception as e:
                    cr = CandidateResult(
                        sample_idx=si, response_text=f"<ERROR: {e}>",
                        parse_error=True, train_passed=0, train_total=len(tasks[ti].train_pairs),
                        train_score=0.0, test_correct=None, cost_usd=0.0, tokens_in=0, tokens_out=0,
                    )
                with progress_lock:
                    results[ti].candidates.append(cr)
                    n_done += 1

        # Round summary
        deployable = sum(1 for r in results if r.best_program_test_correct)
        any_train = sum(1 for r in results if r.any_train_solved)
        total_cand = sum(len(r.candidates) for r in results)
        cost = sum(c.cost_usd for r in results for c in r.candidates)
        print(f"  {round_label} done: deploy {deployable}/{len(tasks)}, any_train {any_train}, "
              f"total_cand {total_cand}, cost ${cost:.3f}, elapsed {(time.time()-t0)/60:.0f}m")

        # Save after each round
        save(results, output_path, args, t0, f"round{round_idx+1}")

    # Final
    elapsed = time.time() - t0
    s = summarize(results)
    lo, hi = wilson(s["deployable_k"], s["n_tasks"])

    print()
    print("=" * 70)
    print(f"FINAL — Direction A evolutionary ({args.rounds} rounds × K={args.k_per_round})")
    print("=" * 70)
    print(f"  Deployable: {s['deployable_k']}/{s['n_tasks']} = {s['deployable_rate']:.1%}  Wilson [{lo:.1%},{hi:.1%}]")
    print(f"  Oracle:     {s['oracle_k']}/{s['n_tasks']} = {s['oracle_rate']:.1%}")
    print(f"  Cost: ${s['total_cost_usd']:.3f}, elapsed {elapsed/60:.0f}m")
    print()

    # Compare to budget-matched cold baseline
    # E2-bis was K=32 cold → 13/30 = 43.3%
    # Our budget is K=24 (3 rounds × 8), which is 75% of K=32
    # E2-bis K=8 (from V5) → 13/30 = 43.3%
    print("Budget comparison:")
    print(f"  Dir A evolutionary (3×K=8 = 24 total): {s['deployable_k']}/{s['n_tasks']} = {s['deployable_rate']:.1%}")
    print(f"  E2-bis cold K=32:                      13/30 = 43.3%")
    print(f"  V5 cold K=8:                           13/30 = 43.3%")
    delta = s["deployable_k"] - 13
    if delta > 0:
        print(f"  ✅ Evolutionary adds +{delta} tasks over cold baseline!")
    elif delta == 0:
        print(f"  ≈ Evolutionary ties cold baseline (0 additional tasks)")
    else:
        print(f"  ❌ Evolutionary is worse than cold baseline by {-delta} tasks")

    save(results, output_path, args, t0, "final")


def save(results, output_path, args, t0, phase):
    s = summarize(results)
    out = {
        "experiment": "dirA_evolutionary",
        "phase": phase,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_tasks": args.n,
        "k_per_round": args.k_per_round,
        "rounds": args.rounds,
        "total_k": args.k_per_round * args.rounds,
        "top_n_attempts": args.top_n_attempts,
        "elapsed_s": time.time() - t0,
        "summary": s,
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
                        "i": c.sample_idx, "train_score": c.train_score,
                        "test_correct": c.test_correct, "parse_error": c.parse_error,
                        "cost_usd": c.cost_usd,
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
