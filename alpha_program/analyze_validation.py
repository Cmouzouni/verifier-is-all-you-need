"""Analyze the validation block: produce the headline cost-correctness Pareto.

Reads all relevant result JSONs from results/alpha_program/ and produces:
  1. A clean text summary with Wilson 95% CIs for each run
  2. The cost-correctness scatter (text-mode for now; matplotlib later)
  3. A side-by-side comparison vs E2-bis and against published frontier numbers

Run after V1 (and ideally V2a, V3) finish.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class RunSummary:
    label: str
    architecture: str
    benchmark: str
    n_tasks: int
    k_correct: int
    rate: float
    ci_lo: float
    ci_hi: float
    total_cost: float
    cost_per_task: float
    notes: str = ""

    @property
    def ci_str(self) -> str:
        return f"[{self.ci_lo:.1%}, {self.ci_hi:.1%}]"


def parse_validation_run(path: Path, k_per_task: int) -> RunSummary | None:
    """Parse a results JSON from run_validation.py (verifier-loop runs)."""
    if not path.exists():
        return None
    data = json.load(open(path))
    per_task = data.get("per_task", [])
    n_complete = sum(1 for t in per_task if len(t.get("candidates", [])) >= k_per_task)
    if n_complete == 0:
        return None
    deployable = sum(
        1 for t in per_task
        if len(t.get("candidates", [])) >= k_per_task and t.get("first_passing_test_correct")
    )
    rate = deployable / n_complete
    lo, hi = wilson_ci(deployable, n_complete)
    cost = data.get("summary", {}).get("total_cost_usd", 0.0)
    return RunSummary(
        label=data.get("label") or path.stem,
        architecture=f"AlphaProgram K={k_per_task}",
        benchmark=f"{data.get('dataset', '?')}/{data.get('split', '?')}",
        n_tasks=n_complete,
        k_correct=deployable,
        rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        total_cost=cost,
        cost_per_task=cost / n_complete if n_complete else 0,
        notes=f"K={k_per_task}, model={data.get('model', '?')}",
    )


def parse_frontier_run(path: Path) -> RunSummary | None:
    """Parse a results JSON from run_frontier_baseline.py (single-shot runs)."""
    if not path.exists():
        return None
    data = json.load(open(path))
    n = data.get("n_tasks_completed", 0)
    if n == 0:
        return None
    k = data.get("n_correct", 0)
    rate = k / n
    lo, hi = wilson_ci(k, n)
    cost = data.get("total_cost_usd", 0.0)
    return RunSummary(
        label=data.get("label") or path.stem,
        architecture="single-shot",
        benchmark=f"{data.get('dataset', '?')}/{data.get('split', '?')}",
        n_tasks=n,
        k_correct=k,
        rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        total_cost=cost,
        cost_per_task=cost / n if n else 0,
        notes=f"model={data.get('model', '?')}, max_tokens={data.get('max_tokens', '?')}",
    )


def parse_e2_bis_qwen22b() -> RunSummary | None:
    """Hand-loaded E2-bis result on training-30 with K=32."""
    path = RESULTS_DIR / "alpha_program" / "e2_verifier_loop_qwen22b.json"
    if not path.exists():
        return None
    return parse_validation_run(path, k_per_task=32)


# Published frontier-model numbers (for the Pareto plot reference points)
PUBLISHED_REFERENCE = [
    # (label, architecture, n, rate, cost/task USD, source)
    ("GPT-4o", "single-shot", 400, 0.07, 0.05, "ARC Prize 2024 leaderboard"),
    ("Claude 3.5 Sonnet", "single-shot", 400, 0.21, 0.10, "ARC Prize 2024 leaderboard"),
    ("DeepSeek-R1", "thinking", 400, 0.16, 0.30, "ARC-AGI public eval"),
    ("o1", "thinking", 400, 0.32, 1.50, "ARC Prize 2024"),
    ("Akyürek 8B + TTT (semi-private)", "TTT", 100, 0.475, None, "arXiv:2411.07279"),
    ("MindsAI ARC Prize 2024 winner", "TTT+ind", 100, 0.555, None, "ARC Prize 2024"),
]


def main():
    output_dir = RESULTS_DIR / "alpha_program"

    runs: list[RunSummary] = []

    # Load all available run results
    e2_bis = parse_e2_bis_qwen22b()
    if e2_bis:
        e2_bis.label = "E2-bis (qwen-22b verifier-loop K=32, training-30)"
        runs.append(e2_bis)

    # V1 stage 1
    for fname, k in [
        ("v1_arc1_eval_qwen22b_n100_k16.json", 16),
        ("v1_arc1_eval_qwen22b_n100.json", 32),
        ("v1_arc1_eval_qwen22b.json", 32),
    ]:
        s = parse_validation_run(output_dir / fname, k_per_task=k)
        if s:
            s.label = f"V1 ({s.label})"
            runs.append(s)
            break  # use the first one we find

    # V3 (prefer the K=16 file, fall back to legacy K=32)
    for fname, k in [
        ("v3_arc2_pilot_qwen22b_k16.json", 16),
        ("v3_arc2_pilot_qwen22b.json", 32),
    ]:
        s = parse_validation_run(output_dir / fname, k_per_task=k)
        if s:
            s.label = f"V3 ({s.label})"
            runs.append(s)

    # V2a (DeepSeek-V3.1) — prefer the fixed-parser run
    for fname in [
        "v2a_deepseek_v31_eval400_fixed.json",
        "v2a_deepseek_v31_eval400_4k.json",
        "v2a_deepseek_v31_eval400.json",
    ]:
        s = parse_frontier_run(output_dir / fname)
        if s:
            s.label = f"V2a ({s.label})"
            runs.append(s)
            break

    print("=" * 100)
    print("ALPHAPROGRAM VALIDATION BLOCK — RESULTS")
    print("=" * 100)
    print()

    if not runs:
        print("No completed runs to analyze yet.")
        return

    print(f"{'Run':60s} {'n':>5}  {'rate':>8}   {'95% CI':>14}   {'$/task':>9}   {'$ total':>8}")
    print("-" * 110)
    for r in sorted(runs, key=lambda x: -x.rate):
        print(f"{r.label[:60]:60s} {r.n_tasks:>5d}  {r.rate:>7.1%}   {r.ci_str:>14}   ${r.cost_per_task:>7.4f}   ${r.total_cost:>6.2f}")

    print()
    print("Published references (NOT same-task, for context):")
    print("-" * 110)
    for label, arch, n, rate, ct, src in PUBLISHED_REFERENCE:
        ct_str = f"${ct:.2f}/task" if ct is not None else "—"
        print(f"  {label:35s} {arch:15s}  n≈{n:>3}  {rate:5.1%}   {ct_str}   ({src})")

    print()
    print("=" * 100)
    print("HEADLINE COMPARISON (cost-per-correct-answer ranking)")
    print("=" * 100)
    # Build cost-per-correct table for all runs (ours + published)
    rows = []
    for r in runs:
        if r.rate > 0:
            rows.append((r.label, r.rate, r.cost_per_task, r.cost_per_task / r.rate, "same-task"))
    for label, arch, n, rate, ct, src in PUBLISHED_REFERENCE:
        if ct is None or rate == 0:
            continue
        rows.append((label, rate, ct, ct / rate, "published"))
    rows.sort(key=lambda x: x[3])  # cheapest cost-per-correct first

    print(f"{'rank':>4}  {'architecture':50s} {'rate':>7}  {'$/task':>9}  {'$/correct':>10}")
    print("-" * 95)
    for i, (label, rate, ct, cpc, kind) in enumerate(rows, 1):
        kind_marker = "★" if kind == "same-task" else " "
        print(f" {i:>3} {kind_marker} {label[:50]:50s} {rate:>6.1%}  ${ct:>7.4f}  ${cpc:>8.3f}")
    print()
    print("  ★ = same-task run (this validation block); other rows are published numbers.")
    print()

    # Highlight V1 specifically as the validated headline
    v1_run = next((r for r in runs if "V1" in r.label and "eval" in r.label.lower()), None)
    if v1_run:
        print("=" * 100)
        print("VALIDATED HEADLINE — AlphaProgram on canonical ARC-AGI-1 public eval")
        print("=" * 100)
        print(f"  {v1_run.label}")
        print(f"  Rate: {v1_run.rate:.1%} (Wilson 95% CI {v1_run.ci_str}), n={v1_run.n_tasks}")
        print(f"  Cost: ${v1_run.cost_per_task:.4f} / task, ${v1_run.total_cost:.2f} total")
        print(f"  Cost-per-correct: ${v1_run.cost_per_task / v1_run.rate:.3f}")
        print()
        for label, arch, n, rate, ct, src in PUBLISHED_REFERENCE:
            if ct is None:
                continue
            delta_pp = (v1_run.rate - rate) * 100
            cpt_ratio = ct / v1_run.cost_per_task if v1_run.cost_per_task > 0 else float("inf")
            cpc_v1 = v1_run.cost_per_task / v1_run.rate if v1_run.rate > 0 else float("inf")
            cpc_pub = ct / rate if rate > 0 else float("inf")
            cpc_ratio = cpc_pub / cpc_v1 if cpc_v1 > 0 else float("inf")
            sign = "+" if delta_pp >= 0 else ""
            print(f"  vs {label:25s}: {sign}{delta_pp:6.1f}pp accuracy | {cpt_ratio:5.1f}× cheaper $/task | {cpc_ratio:5.1f}× cheaper $/correct")


if __name__ == "__main__":
    main()
