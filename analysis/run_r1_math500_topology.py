"""R1: MATH-500 topology validation — majority vote vs LLM synthesizer.

Answers the reviewer's question: does the synthesis node that dominated our
hand-crafted Phase A battery replicate on a standard benchmark?

Design:
  For each of N MATH-500 problems:
    1. Generate K independent proposals (qwen-22b, temperature 0.9)
    2. Extract answer from each proposal (look for \\boxed{} or ANSWER:)
    3. Aggregate two ways:
       (a) Majority vote — most common extracted answer
       (b) LLM synthesizer — a (K+1)-th call where qwen-22b reads all K
           proposals and picks/synthesises the best answer
    4. Compare each to ground truth via normalized string match + sympy

Reports both rates with Wilson CIs.  Saves per-problem JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt
from pathlib import Path
from threading import Lock

# ── path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.client import LLMClient

# ── prompts ─────────────────────────────────────────────────────────────
PROPOSE_SYSTEM = """\
You are an expert mathematician. Solve the given problem step by step.
After your reasoning, state your final answer inside \\boxed{}.
"""

SYNTHESIZE_SYSTEM = """\
You are an expert mathematician acting as a judge.
You will see several candidate solutions to the same math problem.
Carefully evaluate each solution's reasoning and answer.
Pick the best answer (or synthesize a corrected one if all are flawed).
Output ONLY your final answer inside \\boxed{}.
Do NOT repeat the full solutions — just give the answer.
"""

# ── answer extraction & normalization ──────────────────────────────────
_BOXED_RE = re.compile(r"\\boxed\{", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"^.*?</think>", re.DOTALL)


def _find_matching_brace(text: str, start: int) -> int:
    """Find the matching closing brace for an opening brace at `start`."""
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else len(text) - 1


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text, handling nested braces."""
    # Find all \boxed{ positions and take the last one
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    start = last.end()  # position right after the opening {
    end = _find_matching_brace(text, start)
    return text[start:end].strip()


def extract_answer(text: str) -> str:
    """Extract answer from LLM output. Priority: \\boxed{} > ANSWER: > last number."""
    # Strip think tags
    text = _THINK_RE.sub("", text).strip()
    text = _THINK_OPEN_RE.sub("", text).strip()
    if not text:
        return ""

    # 1. \boxed{...}
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # 2. ANSWER: pattern
    for line in text.split("\n"):
        line = line.strip()
        for prefix in ["ANSWER:", "Answer:", "answer:", "FINAL ANSWER:", "Final Answer:",
                        "**ANSWER:**", "**Answer:**"]:
            if line.startswith(prefix):
                val = line[len(prefix):].strip().rstrip(".")
                val = re.sub(r"[\*]", "", val).strip()
                if val:
                    return val

    # 3. "the answer is X"
    m = re.search(r"[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)", text)
    if m:
        return m.group(1).strip()

    # 4. Last \boxed might have been missed (shouldn't happen), try regex
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()

    # 5. Last line heuristic
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line and len(line) < 80 and not line.startswith("#"):
            return line

    return ""


def normalize_answer(ans: str) -> str:
    """Normalize a math answer for comparison.

    Handles: whitespace, LaTeX wrappers, common formatting differences.
    Returns a canonical string for equality comparison.
    """
    if not ans:
        return ""
    s = ans.strip()

    # Remove surrounding $ signs, \text{}, display math wrappers
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s).strip()
    s = re.sub(r"^\\textbf\{(.*)\}$", r"\1", s).strip()
    s = re.sub(r"^\\mathrm\{(.*)\}$", r"\1", s).strip()
    s = re.sub(r"^\\left\s*", "", s)
    s = re.sub(r"\s*\\right$", "", s)

    # Remove trailing period
    s = s.rstrip(".")

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize common LaTeX: \frac{a}{b} -> a/b for simple fracs
    # But keep original too for exact match
    return s


def _try_numeric(s: str) -> float | None:
    """Try to parse a string as a number (int, float, or simple fraction)."""
    s = s.strip()
    # Direct number
    try:
        return float(s)
    except ValueError:
        pass
    # Simple fraction: a/b
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # LaTeX fraction: \frac{a}{b}  or  \dfrac{a}{b}
    m = re.match(r"^\\d?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer.

    Strategy:
      1. Exact normalized string match
      2. Numeric match (within tolerance)
      3. Sympy symbolic match (if available)
    """
    np = normalize_answer(pred)
    ng = normalize_answer(gold)

    if not np or not ng:
        return False

    # 1. Exact string match
    if np == ng:
        return True

    # 1b. Case-insensitive for text answers
    if np.lower() == ng.lower():
        return True

    # 2. Numeric match
    nv_pred = _try_numeric(np)
    nv_gold = _try_numeric(ng)
    if nv_pred is not None and nv_gold is not None:
        if abs(nv_pred - nv_gold) < 1e-6:
            return True

    # 3. Try sympy for symbolic equivalence
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, Eq, N
        sp = parse_latex(np)
        sg = parse_latex(ng)
        if simplify(sp - sg) == 0:
            return True
        # Also try numeric evaluation
        try:
            diff = abs(complex(N(sp - sg)))
            if diff < 1e-6:
                return True
        except (TypeError, ValueError):
            pass
    except Exception:
        pass

    return False


# ── Wilson CI ──────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="R1: MATH-500 majority vote vs LLM synthesizer")
    parser.add_argument("--n", type=int, default=500, help="Number of problems to run (default: 500)")
    parser.add_argument("--k", type=int, default=4, help="Number of proposals per problem (default: 4)")
    parser.add_argument("--workers", type=int, default=16, help="Max concurrent API calls (default: 16)")
    parser.add_argument("--output", type=str, default="r1_math500_topology.json", help="Output filename")
    parser.add_argument("--model", type=str, default="qwen-22b", help="Model key (default: qwen-22b)")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"R1: MATH-500 topology validation")
    print(f"  Model: {args.model}, K={args.k}, N={args.n}, T={args.temperature}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")
    print()

    # ── Load dataset ──────────────────────────────────────────────────
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.n < len(ds):
        # Deterministic subsample
        import random
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(ds)), args.n))
        problems = [ds[i] for i in indices]
    else:
        problems = [ds[i] for i in range(len(ds))]
    print(f"  Loaded {len(problems)} problems from MATH-500\n")

    # ── LLM client ────────────────────────────────────────────────────
    client = LLMClient(model_key=args.model, seed=args.seed)

    # ── Per-problem processing ────────────────────────────────────────
    results = []
    lock = Lock()
    progress = {"n_done": 0}
    t0 = time.time()

    def process_problem(idx: int, item: dict) -> dict:
        """Generate K proposals, then majority-vote and synthesizer aggregation."""
        problem_text = item["problem"]
        gold_answer = item["answer"]
        uid = item.get("unique_id", f"problem_{idx}")

        # Step 1: Generate K proposals (sequential within problem to avoid overloading)
        proposals = []
        extracted_answers = []
        for ki in range(args.k):
            resp = client.generate(
                system_prompt=PROPOSE_SYSTEM,
                user_prompt=f"Problem:\n{problem_text}",
                temperature=args.temperature,
                max_tokens=2048,
            )
            proposals.append(resp.content)
            ans = extract_answer(resp.content)
            extracted_answers.append(ans)

        # Step 2a: Majority vote
        # Normalize answers for voting, then pick the most common
        norm_answers = [normalize_answer(a) for a in extracted_answers]
        # Group by normalized form
        vote_counts = Counter(norm_answers)
        # Remove empty votes
        vote_counts.pop("", None)
        if vote_counts:
            majority_answer = vote_counts.most_common(1)[0][0]
            # Map back to the original extracted answer for the most common normalized form
            for ea, na in zip(extracted_answers, norm_answers):
                if na == majority_answer:
                    majority_raw = ea
                    break
            else:
                majority_raw = majority_answer
        else:
            majority_answer = ""
            majority_raw = ""

        majority_correct = answers_match(majority_raw, gold_answer)

        # Step 2b: LLM synthesizer
        # Build the prompt showing all K proposals
        proposals_text = ""
        for ki, (prop, ans) in enumerate(zip(proposals, extracted_answers)):
            # Truncate very long proposals to keep context manageable
            prop_short = prop[:1500] if len(prop) > 1500 else prop
            proposals_text += f"\n--- Candidate {ki+1} ---\n{prop_short}\n"

        synth_resp = client.generate(
            system_prompt=SYNTHESIZE_SYSTEM,
            user_prompt=f"Problem:\n{problem_text}\n\nCandidate solutions:{proposals_text}",
            temperature=0.3,  # Low temperature for judge
            max_tokens=1024,
        )
        synth_answer = extract_answer(synth_resp.content)
        synth_correct = answers_match(synth_answer, gold_answer)

        result = {
            "idx": idx,
            "unique_id": uid,
            "subject": item.get("subject", ""),
            "level": item.get("level", 0),
            "gold_answer": gold_answer,
            "extracted_answers": extracted_answers,
            "norm_answers": norm_answers,
            "vote_counts": dict(vote_counts),
            "majority_answer": majority_answer,
            "majority_correct": majority_correct,
            "synth_answer": synth_answer,
            "synth_correct": synth_correct,
            # Per-proposal correctness (any-correct = pass@k)
            "per_proposal_correct": [answers_match(ea, gold_answer) for ea in extracted_answers],
        }

        with lock:
            progress["n_done"] += 1
            nd = progress["n_done"]
            mv = "V" if majority_correct else "."
            sv = "S" if synth_correct else "."
            any_c = any(result["per_proposal_correct"])
            ac = "A" if any_c else "."
            print(f"  [{nd:3d}/{len(problems)}] {uid:40s}  MV={mv} Synth={sv} Any={ac}  "
                  f"gold={gold_answer[:30]:30s}  mv={majority_answer[:30]:30s}  "
                  f"synth={synth_answer[:30]:30s}", flush=True)

        return result

    # Run problems in parallel (each problem is K+1 sequential LLM calls,
    # but different problems run concurrently)
    # Limit concurrency: each problem makes K+1 calls, so effective API concurrency ~ workers*(K+1)
    effective_workers = max(1, args.workers // (args.k + 1))
    print(f"  Effective problem-level parallelism: {effective_workers}")
    print(f"  Expected API concurrency: ~{effective_workers} (sequential within problem)\n")

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(process_problem, i, item): i for i, item in enumerate(problems)}
        for f in as_completed(futures):
            try:
                result = f.result()
                results.append(result)
            except Exception as e:
                idx = futures[f]
                print(f"  [ERROR] Problem {idx}: {e}", flush=True)
                results.append({"idx": idx, "error": str(e),
                                "majority_correct": False, "synth_correct": False,
                                "per_proposal_correct": [False] * args.k})

    elapsed = time.time() - t0

    # ── Sort results by index ─────────────────────────────────────────
    results.sort(key=lambda r: r.get("idx", 0))

    # ── Compute aggregates ────────────────────────────────────────────
    n_total = len(results)
    n_mv_correct = sum(1 for r in results if r["majority_correct"])
    n_synth_correct = sum(1 for r in results if r["synth_correct"])
    n_any_correct = sum(1 for r in results if any(r["per_proposal_correct"]))

    mv_rate = n_mv_correct / n_total if n_total > 0 else 0
    synth_rate = n_synth_correct / n_total if n_total > 0 else 0
    any_rate = n_any_correct / n_total if n_total > 0 else 0

    mv_ci = wilson_ci(n_mv_correct, n_total)
    synth_ci = wilson_ci(n_synth_correct, n_total)
    any_ci = wilson_ci(n_any_correct, n_total)

    # Per-proposal pass@1 (average across proposals)
    all_p1 = [answers_match(ea, r["gold_answer"])
              for r in results if "extracted_answers" in r
              for ea in r.get("extracted_answers", [])]
    p1_rate = sum(all_p1) / len(all_p1) if all_p1 else 0

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"R1 FINAL: MATH-500 topology validation ({args.model}, K={args.k})")
    print(f"{'='*70}")
    print(f"  Problems:        {n_total}")
    print(f"  pass@1 (avg):    {p1_rate:.1%}")
    print(f"  pass@K (any):    {n_any_correct}/{n_total} = {any_rate:.1%}  [{any_ci[0]:.1%}, {any_ci[1]:.1%}]")
    print(f"  Majority vote:   {n_mv_correct}/{n_total} = {mv_rate:.1%}  [{mv_ci[0]:.1%}, {mv_ci[1]:.1%}]")
    print(f"  LLM synthesizer: {n_synth_correct}/{n_total} = {synth_rate:.1%}  [{synth_ci[0]:.1%}, {synth_ci[1]:.1%}]")
    print(f"  Delta (S - MV):  {(synth_rate - mv_rate):+.1%}")
    print(f"  Elapsed:         {elapsed:.0f}s")
    print(f"  Total cost:      ${client.tracker.total_cost:.2f}")
    print(f"  API calls:       {client.tracker.n_calls}")

    # Breakdown by difficulty level
    for level in sorted(set(r.get("level", 0) for r in results)):
        level_results = [r for r in results if r.get("level") == level]
        nl = len(level_results)
        if nl == 0:
            continue
        mv_l = sum(1 for r in level_results if r["majority_correct"])
        sy_l = sum(1 for r in level_results if r["synth_correct"])
        print(f"    Level {level}: MV={mv_l}/{nl} ({mv_l/nl:.0%})  Synth={sy_l}/{nl} ({sy_l/nl:.0%})  "
              f"Delta={(sy_l/nl - mv_l/nl):+.0%}  (n={nl})")

    # ── Save results ──────────────────────────────────────────────────
    outdir = ROOT / "results" / "alpha_program"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.output

    output = {
        "experiment": "r1_math500_topology",
        "model": args.model,
        "k": args.k,
        "n": n_total,
        "temperature": args.temperature,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "total_cost_usd": client.tracker.total_cost,
        "n_api_calls": client.tracker.n_calls,
        "summary": {
            "pass_at_1_avg": p1_rate,
            "pass_at_k_any": any_rate,
            "majority_vote_rate": mv_rate,
            "majority_vote_ci": list(mv_ci),
            "synth_rate": synth_rate,
            "synth_ci": list(synth_ci),
            "delta_synth_minus_mv": synth_rate - mv_rate,
        },
        "per_problem": results,
    }

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
