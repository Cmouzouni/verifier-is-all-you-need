"""E2-bis — Verifier-in-loop best-of-N with qwen-22b (the contingency).

E2 with qwen-7b failed at 6.7% deployable rate (vs 35% pass criterion).
The diagnostic was unambiguous: 98.5% of qwen-7b's K=960 candidates
passed 0 train pairs. The bottleneck was generation, not verification.

This run upgrades the proposer to qwen-22b (Qwen3-235B-A22B), which is
~3-4x stronger on code generation while being CHEAPER per token for
this workload (output is short, qwen-22b has lower input pricing).

If qwen-22b clears 35%, the architecture is validated and we proceed
to E3. If it doesn't, we go to qwen-coder-32b (a code specialist) or
re-engineer the DSL/prompt.

Cost estimate: ~$1.00 (4.3M input tokens at $0.20/M + 234K output at $0.60/M).
"""
from __future__ import annotations

import alpha_program.exp_e2_verifier_loop as base
from alpha_program.exp_e2_verifier_loop import main

# Override the model key
base.MODEL_KEY = "qwen-22b"
base.PASS_THRESHOLD = 0.35


def save_results_qwen22b(results, output_dir, t0):
    """Override save path to keep qwen-7b results separate."""
    import json
    from alpha_program.exp_e2_verifier_loop import summarize, K_PROGRAMS, TEMPERATURE, MAX_TOKENS
    import time
    out = {
        "experiment": "e2_verifier_loop_qwen22b",
        "model": "qwen-22b",
        "n_tasks_planned": base.N_TASKS,
        "n_tasks_completed": len(results),
        "k_programs": K_PROGRAMS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "pass_threshold": base.PASS_THRESHOLD,
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
    json.dump(out, open(output_dir / "e2_verifier_loop_qwen22b.json", "w"), indent=2, default=str)


# Monkey-patch the save function to use a different filename
base.save_results = save_results_qwen22b


if __name__ == "__main__":
    main()
