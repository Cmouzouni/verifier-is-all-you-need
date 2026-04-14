"""Symbolic verifier — runs proposer-generated `transform(grid)` programs
against ARC training pairs in a sandboxed namespace and reports a score.

This is the value head of the AlphaProgram architecture (the cheap value head
in the AlphaZero analogy). Cost is microseconds per program. No LLM in this
loop. The verifiability boundary doesn't bind here because verification is
execution, not LLM judgment.

Public API:
    extract_program(text: str) -> str | None
        Pull the proposer's `def transform(grid):` block out of an LLM response.
    run_program(program_src: str, grid: Grid, timeout_s: float) -> RunResult
        Execute one program on one grid, in a sandbox, with timeout.
    score_program(program_src: str, train_pairs, timeout_s: float) -> ScoreResult
        Run a program against all train_pairs, return per-pair pass/fail and total score.
"""
from __future__ import annotations

import multiprocessing as mp
import re
import signal
import traceback
from dataclasses import dataclass, field
from typing import Any

from .dsl import DSL_NAMESPACE, Grid


# ───── extracting the program from a model response ───────────────────────

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"(def\s+transform\s*\(.*?\)\s*:.*?)(?=\n(?:def|\Z|\Z))", re.DOTALL)


def extract_program(text: str) -> str | None:
    """Pull a `def transform(grid):` Python block out of an LLM response.

    Strategy:
      1. Look for the last ``` fenced block that defines `transform`.
      2. Otherwise, look for any `def transform(...):` block in raw text.
      3. Returns the source as a string, or None if no candidate found.
    """
    if not text:
        return None
    # strip <think> blocks (qwen3 thinking-mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # try fenced blocks first
    for block in reversed(_FENCE_RE.findall(text)):
        if "def transform" in block:
            return block.strip()
    # any raw def transform anywhere
    if "def transform" in text:
        # take from the first def to end of text and strip trailing prose
        idx = text.find("def transform")
        candidate = text[idx:]
        # cut at the next code-fence end if present
        if "```" in candidate:
            candidate = candidate.split("```", 1)[0]
        return candidate.strip()
    return None


# ───── sandbox execution ──────────────────────────────────────────────────

@dataclass
class RunResult:
    ok: bool
    output: Grid | None = None
    error: str | None = None
    elapsed_ms: float = 0.0


def _build_sandbox_globals() -> dict[str, Any]:
    """A restricted globals dict containing only DSL helpers + safe builtins.

    NOTE: this does NOT prevent a malicious program from doing harmful things
    in the same process — Python sandboxing is hard. We rely on the
    multiprocessing wrapper to provide process isolation + timeout. The
    namespace restriction is for proposer-clarity, not security.
    """
    g = {"__builtins__": {}}
    g.update(DSL_NAMESPACE)
    return g


def _run_in_subprocess(program_src: str, input_grid: Grid, queue: mp.Queue) -> None:
    """Worker: compile program, run transform(input_grid), put result on queue."""
    try:
        sandbox = _build_sandbox_globals()
        compiled = compile(program_src, "<proposer>", "exec")
        exec(compiled, sandbox)
        if "transform" not in sandbox:
            queue.put(("error", "no `transform` function defined"))
            return
        result = sandbox["transform"]([row[:] for row in input_grid])
        # validate it's a 2-D int grid
        if not isinstance(result, list):
            queue.put(("error", f"transform returned {type(result).__name__}, expected list"))
            return
        if not result:
            queue.put(("ok", []))
            return
        if not all(isinstance(r, list) for r in result):
            queue.put(("error", "transform did not return list of lists"))
            return
        if not all(all(isinstance(c, int) and 0 <= c <= 9 for c in row) for row in result):
            queue.put(("error", "transform returned non-integer or out-of-range values"))
            return
        # check rectangular
        if len({len(r) for r in result}) > 1:
            queue.put(("error", "transform returned non-rectangular grid"))
            return
        queue.put(("ok", result))
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        # keep just the most relevant lines
        msg = f"{type(e).__name__}: {e}"
        queue.put(("error", msg))


def run_program(program_src: str, input_grid: Grid, timeout_s: float = 2.0) -> RunResult:
    """Execute one program on one input grid in a subprocess with a hard timeout."""
    import time
    t0 = time.time()
    ctx = mp.get_context("fork")  # cheap on macOS/Linux
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_run_in_subprocess, args=(program_src, input_grid, q))
    p.start()
    p.join(timeout_s)
    elapsed_ms = (time.time() - t0) * 1000
    if p.is_alive():
        p.terminate()
        p.join(0.2)
        if p.is_alive():
            p.kill()
        return RunResult(ok=False, error=f"timeout after {timeout_s}s", elapsed_ms=elapsed_ms)
    if q.empty():
        return RunResult(ok=False, error="subprocess died without result", elapsed_ms=elapsed_ms)
    tag, payload = q.get()
    if tag == "ok":
        return RunResult(ok=True, output=payload, elapsed_ms=elapsed_ms)
    return RunResult(ok=False, error=payload, elapsed_ms=elapsed_ms)


# ───── scoring against ARC train pairs ────────────────────────────────────

@dataclass
class ScoreResult:
    n_train: int
    n_passed: int
    score: float  # n_passed / n_train
    per_pair: list[dict] = field(default_factory=list)
    error: str | None = None
    total_elapsed_ms: float = 0.0
    test_output: Grid | None = None  # what the program produces on test_input
    parse_error: bool = False  # set when extract_program returned None


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for r1, r2 in zip(a, b):
        if len(r1) != len(r2):
            return False
        if any(x != y for x, y in zip(r1, r2)):
            return False
    return True


def score_program(
    program_src: str,
    train_pairs: list[tuple[Grid, Grid]],
    test_input: Grid | None = None,
    timeout_s: float = 2.0,
) -> ScoreResult:
    """Run a program against all train pairs and (optionally) the test input.

    Returns ScoreResult with per-pair pass/fail, total score, and the
    program's prediction on `test_input` (if given).
    """
    per_pair = []
    n_passed = 0
    total_elapsed = 0.0
    fatal_error = None

    for i, (inp, expected) in enumerate(train_pairs):
        r = run_program(program_src, inp, timeout_s=timeout_s)
        total_elapsed += r.elapsed_ms
        if not r.ok:
            per_pair.append({"i": i, "passed": False, "error": r.error})
            continue
        if r.output is None:
            per_pair.append({"i": i, "passed": False, "error": "no output"})
            continue
        passed = _grids_equal(r.output, expected)
        per_pair.append({
            "i": i, "passed": passed,
            "expected_shape": (len(expected), len(expected[0]) if expected else 0),
            "got_shape": (len(r.output), len(r.output[0]) if r.output else 0),
        })
        if passed:
            n_passed += 1

    test_output = None
    if test_input is not None:
        r = run_program(program_src, test_input, timeout_s=timeout_s)
        total_elapsed += r.elapsed_ms
        if r.ok:
            test_output = r.output

    return ScoreResult(
        n_train=len(train_pairs),
        n_passed=n_passed,
        score=n_passed / len(train_pairs) if train_pairs else 0.0,
        per_pair=per_pair,
        error=fatal_error,
        total_elapsed_ms=total_elapsed,
        test_output=test_output,
    )


def score_response(
    response_text: str,
    train_pairs: list[tuple[Grid, Grid]],
    test_input: Grid | None = None,
    timeout_s: float = 2.0,
) -> ScoreResult:
    """Convenience: extract a program from an LLM response and score it."""
    src = extract_program(response_text)
    if src is None:
        return ScoreResult(
            n_train=len(train_pairs),
            n_passed=0,
            score=0.0,
            error="no `transform` function found in response",
            parse_error=True,
        )
    return score_program(src, train_pairs, test_input=test_input, timeout_s=timeout_s)
