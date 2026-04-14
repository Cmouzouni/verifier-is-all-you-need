"""AIME executor — runs proposer-generated `def solve():` programs in a
sandbox and extracts an integer answer.

Unlike the ARC verifier, AIME has no train pairs to filter on. The
"verifier" here is just "code runs and produces a sane integer." The
actual selection happens via majority vote across K samples (the runner's
job, not this module).

The proposer is allowed to import math, fractions, itertools, sympy
(yes, sympy is allowed for AIME — that's the whole point). Other
imports are blocked.
"""
from __future__ import annotations

import multiprocessing as mp
import re
import time
import traceback
from dataclasses import dataclass


# ───── extracting the program ──────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str | None:
    """Pull a `def solve():` Python block out of an LLM response."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Prefer the LAST fenced block that defines `solve`
    blocks = _FENCE_RE.findall(text)
    for block in reversed(blocks):
        if "def solve" in block:
            return block.strip()
    if "def solve" in text:
        idx = text.find("def solve")
        candidate = text[idx:]
        if "```" in candidate:
            candidate = candidate.split("```", 1)[0]
        return candidate.strip()
    return None


# ───── sandbox ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    ok: bool
    answer: int | None = None
    error: str | None = None
    elapsed_ms: float = 0.0


# Allowed top-level imports for AIME proposer code. sympy is the
# big one — these are math problems, the model needs symbolic manipulation.
ALLOWED_IMPORTS = {
    "math", "fractions", "decimal", "itertools", "functools", "collections",
    "sympy", "numpy", "statistics", "random", "operator", "bisect",
    "heapq", "re",
}


def _restricted_import(name, *args, **kwargs):
    """Import shim that only allows the whitelist."""
    top = name.split(".")[0]
    if top not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in this sandbox")
    import importlib
    return importlib.import_module(name)


def _build_sandbox_globals() -> dict:
    """Restricted globals for proposer code execution."""
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "bytes": bytes,
        "chr": chr, "complex": complex, "dict": dict, "divmod": divmod,
        "enumerate": enumerate, "filter": filter, "float": float,
        "frozenset": frozenset, "hash": hash, "hex": hex, "int": int,
        "isinstance": isinstance, "issubclass": issubclass, "iter": iter,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "next": next, "object": object, "oct": oct, "ord": ord, "pow": pow,
        "print": lambda *a, **k: None,  # silenced
        "range": range, "repr": repr, "reversed": reversed, "round": round,
        "set": set, "slice": slice, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "type": type, "zip": zip,
        "True": True, "False": False, "None": None,
        "__import__": _restricted_import,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "ZeroDivisionError": ZeroDivisionError,
    }
    return {"__builtins__": safe_builtins}


def _run_in_subprocess(program_src: str, queue: mp.Queue) -> None:
    """Worker: compile, exec, call solve(), put answer on queue."""
    try:
        sandbox = _build_sandbox_globals()
        compiled = compile(program_src, "<aime_proposer>", "exec")
        exec(compiled, sandbox)
        if "solve" not in sandbox:
            queue.put(("error", "no `solve` function defined"))
            return
        result = sandbox["solve"]()
        # Coerce to integer if possible
        if isinstance(result, bool):
            queue.put(("error", "solve returned bool, expected int"))
            return
        try:
            ans = int(result)
        except (TypeError, ValueError):
            queue.put(("error", f"solve returned {type(result).__name__}, not coercible to int: {str(result)[:80]}"))
            return
        # AIME answers are in [0, 999]
        if ans < 0 or ans > 999:
            queue.put(("error", f"solve returned {ans}, out of AIME range [0, 999]"))
            return
        queue.put(("ok", ans))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        queue.put(("error", msg[:200]))


def run_program(program_src: str, timeout_s: float = 10.0) -> RunResult:
    """Execute a proposer's solve() program with a hard timeout."""
    t0 = time.time()
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_run_in_subprocess, args=(program_src, q))
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
        return RunResult(ok=True, answer=payload, elapsed_ms=elapsed_ms)
    return RunResult(ok=False, error=payload, elapsed_ms=elapsed_ms)


def run_response(response_text: str, timeout_s: float = 10.0) -> RunResult:
    """Convenience: extract a program from an LLM response and run it."""
    src = extract_program(response_text)
    if src is None:
        return RunResult(ok=False, error="no solve() function found")
    return run_program(src, timeout_s=timeout_s)
