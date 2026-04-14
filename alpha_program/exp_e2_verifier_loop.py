"""E2 — Verifier-in-loop best-of-N pilot (THE FLOOR).

The decisive first AlphaProgram experiment. qwen-7b proposes K programs per
ARC task, the symbolic verifier runs each against all training pairs, and
we report the per-attempt and best-of-N rates on the same 30-task ARC
battery used by pivot-a (so the comparison is direct).

Pass criterion: ≥35% on the 30 tasks (vs pivot-a 22.7% qwen-22b single-shot
baseline / 33.3% qwen-22b 8-prop majority vote ceiling). Hitting 35% with
qwen-7b — a model 3× cheaper than qwen-22b — is the make-or-break signal
that the AlphaZero template transfers to ARC.

Reports:
  - per-attempt rate (was the proposed program correct on test?)
  - any-passing-on-train rate (did at least one of K programs pass all train?)
  - best-of-K test correctness (when at least one passed train, is its test output right?)
  - mean fraction of train pairs passed by the best program per task

Cost (from PLAN):
  30 tasks × 32 programs × qwen-7b ≈ $0.96
  + K-sweep + retries = budget $5
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks
from alpha_program.dsl import DSL_DOC
from alpha_program.verifier import score_response, score_program, ScoreResult


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

N_TASKS = 30
K_PROGRAMS = 32       # samples per task
SEED_BASE = 42        # reproducibility
TEMPERATURE = 0.9     # high diversity for sampling
MAX_TOKENS = 1024
MODEL_KEY = "qwen-7b"
WORKERS = 8           # API concurrency
TIMEOUT_S = 2.0       # verifier per-program timeout
PASS_THRESHOLD = 0.35 # decision-point pass criterion


# ════════════════════════════════════════════════════════════════════════
# Prompt
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are a Python programmer solving ARC-AGI visual-reasoning puzzles.
Each puzzle gives you a few input/output grid pairs. Your job is to write a
Python function `transform(grid)` that takes a 2-D list of integers (the input
grid) and returns the transformed output grid.

The colors in ARC grids are integers 0-9.

You may use a small library of helper functions:

{DSL_DOC}

CRITICAL RULES:
- Define exactly one function: `def transform(grid):` returning `list[list[int]]`.
- Wrap your code in a single ```python fenced block.
- Do NOT import anything. Do NOT use eval, exec, file I/O, networking, or any side effect.
- Keep the function pure: no global state, no print, no input.
- The function should work on a NEW input grid of the same problem family, not just the examples.
"""


def format_task_prompt(train_pairs, test_input) -> str:
    """Format an ARC task as a Python-programmer prompt."""
    lines = ["Here are the training input/output pairs for one ARC puzzle:\n"]
    for i, (inp, out) in enumerate(train_pairs):
        lines.append(f"# Example {i+1}")
        lines.append(f"input  = {inp!r}")
        lines.append(f"output = {out!r}")
        lines.append("")
    lines.append("# The held-out test input (your transform must work on it):")
    lines.append(f"test_input = {test_input!r}")
    lines.append("")
    lines.append("Write the `def transform(grid):` function that maps input -> output for ALL of the examples above. It will be executed on each training input and graded by exact-match against the training output. Then it will be executed on the test_input.")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Per-task experiment
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateResult:
    sample_idx: int
    response_text: str
    parse_error: bool
    train_passed: int
    train_total: int
    train_score: float
    test_correct: bool | None  # None if program errored before test
    cost_usd: float
    tokens_in: int
    tokens_out: int


@dataclass
class TaskResult:
    task_id: str
    n_train_pairs: int
    candidates: list[CandidateResult] = field(default_factory=list)

    @property
    def best_train_score(self) -> float:
        return max((c.train_score for c in self.candidates), default=0.0)

    @property
    def any_train_solved(self) -> bool:
        return any(c.train_score == 1.0 for c in self.candidates)

    @property
    def best_program_test_correct(self) -> bool:
        """Of the candidates that passed all train pairs, is the FIRST one
        correct on the test? This is the deployable "verifier-in-loop" answer:
        we cannot peek at test, so we tiebreak by "first program that passed
        all train"."""
        for c in self.candidates:
            if c.train_score == 1.0:
                return bool(c.test_correct)
        return False

    @property
    def per_attempt_rate(self) -> float:
        """Fraction of K samples whose program is correct on test_input."""
        if not self.candidates:
            return 0.0
        return sum(1 for c in self.candidates if c.test_correct) / len(self.candidates)

    @property
    def parse_error_rate(self) -> float:
        if not self.candidates:
            return 0.0
        return sum(1 for c in self.candidates if c.parse_error) / len(self.candidates)


def run_one_sample(
    client: LLMClient,
    task,
    sample_idx: int,
) -> CandidateResult:
    """Generate one program for one task and score it."""
    user = format_task_prompt(task.train_pairs, task.test_input)
    # vary seed across samples to encourage diversity
    client.seed = SEED_BASE + sample_idx
    try:
        resp = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
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
        timeout_s=TIMEOUT_S,
    )

    # Check test correctness against ground truth
    test_correct = None
    if sr.test_output is not None:
        from alpha_program.verifier import _grids_equal
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


def run_one_task(
    client: LLMClient,
    task,
    k: int,
) -> TaskResult:
    """Run K parallel samples for one task."""
    result = TaskResult(task_id=task.id, n_train_pairs=len(task.train_pairs))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(run_one_sample, client, task, i): i
            for i in range(k)
        }
        for f in as_completed(futures):
            try:
                cr = f.result()
                result.candidates.append(cr)
            except Exception as e:
                idx = futures[f]
                result.candidates.append(CandidateResult(
                    sample_idx=idx,
                    response_text=f"<RUNNER_ERROR: {e}>",
                    parse_error=True,
                    train_passed=0,
                    train_total=len(task.train_pairs),
                    train_score=0.0,
                    test_correct=None,
                    cost_usd=0.0,
                    tokens_in=0,
                    tokens_out=0,
                ))
    result.candidates.sort(key=lambda c: c.sample_idx)
    return result


# ════════════════════════════════════════════════════════════════════════
# Aggregate metrics
# ════════════════════════════════════════════════════════════════════════

def summarize(results: list[TaskResult]) -> dict:
    n = len(results)
    n_per_attempt = sum(r.per_attempt_rate for r in results)
    per_attempt_rate = sum(
        sum(1 for c in r.candidates if c.test_correct)
        for r in results
    ) / sum(len(r.candidates) for r in results) if results else 0.0

    # The deployable metric: pick "first program that passes all train pairs"
    deployable_correct = sum(1 for r in results if r.best_program_test_correct)
    deployable_rate = deployable_correct / n if n else 0.0

    # The oracle ceiling: was ANY of the K programs correct on test?
    oracle_correct = sum(
        1 for r in results
        if any(c.test_correct for c in r.candidates)
    )
    oracle_rate = oracle_correct / n if n else 0.0

    # Did the proposer manage to write a program that solved all train pairs?
    any_train_solved = sum(1 for r in results if r.any_train_solved)
    any_train_solved_rate = any_train_solved / n if n else 0.0

    total_cost = sum(c.cost_usd for r in results for c in r.candidates)
    total_tokens_in = sum(c.tokens_in for r in results for c in r.candidates)
    total_tokens_out = sum(c.tokens_out for r in results for c in r.candidates)
    parse_error_rate = sum(
        sum(1 for c in r.candidates if c.parse_error) for r in results
    ) / sum(len(r.candidates) for r in results) if results else 0.0

    return {
        "n_tasks": n,
        "n_samples_per_task": K_PROGRAMS,
        "per_attempt_rate": per_attempt_rate,
        "deployable_rate": deployable_rate,
        "deployable_k": deployable_correct,
        "oracle_rate": oracle_rate,
        "oracle_k": oracle_correct,
        "any_train_solved_rate": any_train_solved_rate,
        "any_train_solved_k": any_train_solved,
        "total_cost_usd": total_cost,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "parse_error_rate": parse_error_rate,
    }


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("E2 — Verifier-in-loop best-of-N (THE FLOOR)")
    print("=" * 70)
    print(f"  Model: {MODEL_KEY}")
    print(f"  Tasks: {N_TASKS}")
    print(f"  Samples per task (K): {K_PROGRAMS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Pass threshold: {PASS_THRESHOLD:.0%}")
    print()

    tasks = load_arc_tasks(n=N_TASKS, seed=SEED_BASE)
    print(f"Loaded {len(tasks)} ARC tasks (seed={SEED_BASE})")

    client = LLMClient(MODEL_KEY, seed=SEED_BASE)

    output_dir = RESULTS_DIR / "alpha_program"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []
    t0 = time.time()

    for i, task in enumerate(tasks):
        t_task = time.time()
        print(f"\n[{i+1}/{len(tasks)}] {task.id} ({len(task.train_pairs)} train pairs)", flush=True)
        tr = run_one_task(client, task, K_PROGRAMS)
        results.append(tr)
        elapsed = time.time() - t_task
        print(f"  per_attempt={tr.per_attempt_rate:.0%}  best_train={tr.best_train_score:.0%}  "
              f"any_train_solved={tr.any_train_solved}  test_correct(first-passing)={tr.best_program_test_correct}  "
              f"parse_err={tr.parse_error_rate:.0%}  ({elapsed:.0f}s)")

        # incremental save
        save_results(results, output_dir, t0)

    elapsed = time.time() - t0
    summary = summarize(results)

    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)
    print(f"  per-attempt rate         : {summary['per_attempt_rate']:7.1%}")
    print(f"  deployable (first-passing): {summary['deployable_k']:>3d}/{summary['n_tasks']} = {summary['deployable_rate']:7.1%}")
    print(f"  oracle (any-correct)     : {summary['oracle_k']:>3d}/{summary['n_tasks']} = {summary['oracle_rate']:7.1%}")
    print(f"  any-train-solved         : {summary['any_train_solved_k']:>3d}/{summary['n_tasks']} = {summary['any_train_solved_rate']:7.1%}")
    print(f"  parse error rate         : {summary['parse_error_rate']:7.1%}")
    print(f"  total cost               : ${summary['total_cost_usd']:.4f}")
    print(f"  tokens (in / out)        : {summary['total_tokens_in']:,} / {summary['total_tokens_out']:,}")
    print(f"  elapsed                  : {elapsed:.0f}s")
    print()

    # Decision point
    print("─" * 70)
    if summary['deployable_rate'] >= PASS_THRESHOLD:
        print(f"✅ E2 PASSES: deployable rate {summary['deployable_rate']:.1%} ≥ {PASS_THRESHOLD:.0%}")
        print("   Bet is LIVE. Proceed to E3 (critic loop).")
    else:
        print(f"❌ E2 FAILS: deployable rate {summary['deployable_rate']:.1%} < {PASS_THRESHOLD:.0%}")
        print("   Decision needed: pivot to bigger proposer (qwen-22b/coder), or to E8/E9 first.")
    print("─" * 70)


def save_results(results: list[TaskResult], output_dir: Path, t0: float) -> None:
    out = {
        "experiment": "e2_verifier_loop",
        "model": MODEL_KEY,
        "n_tasks_planned": N_TASKS,
        "n_tasks_completed": len(results),
        "k_programs": K_PROGRAMS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "pass_threshold": PASS_THRESHOLD,
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
    json.dump(out, open(output_dir / "e2_verifier_loop.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
