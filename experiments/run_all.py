"""Master runner: execute ALL Phase A experiments.

Usage:
    python -m experiments.run_all --dry-run    # cost estimate only
    python -m experiments.run_all              # run everything

All experiments use Qwen model family exclusively.
Cost tracking is built into every experiment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR


def main():
    dry_run = "--dry-run" in sys.argv
    t_start = time.time()

    print("=" * 70)
    print("PIVOT-A: FULL EXPERIMENT SUITE")
    print(f"Mode: {'DRY RUN (cost estimate)' if dry_run else 'LIVE EXECUTION'}")
    print("Models: Qwen family only (7B, 22B-active, 17B-frontier)")
    print("=" * 70)

    results = {}
    costs = {}

    # ── A.1: γ Response Validation ─────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("EXPERIMENT A.1: γ RESPONSE VALIDATION")
    print("Does diversity increase monotonically with γ?")
    print("=" * 70)
    from experiments.exp_a1_gamma import run_experiment as run_a1
    results["A.1"] = run_a1(dry_run=dry_run)

    # ── A.2: Information Shielding ─────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("EXPERIMENT A.2: INFORMATION SHIELDING")
    print("Does shielding Insight agents prevent anchoring on planted wrong answers?")
    print("=" * 70)
    from experiments.exp_a2_shielding import run_experiment as run_a2
    results["A.2"] = run_a2(dry_run=dry_run)

    # ── A.6: Game Correctness ──────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("EXPERIMENT A.6: GAME CORRECTNESS")
    print("Does game beat single agent and majority vote at matched cost?")
    print("=" * 70)
    from experiments.exp_a6_correctness import run_experiment as run_a6
    results["A.6"] = run_a6(dry_run=dry_run)

    # ── Final Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n\n" + "=" * 70)
    print("FULL SUITE COMPLETE")
    print(f"Total wall time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")

    if not dry_run:
        # Collect all costs from saved summaries
        total_cost = 0.0
        for exp_id in ["exp_a1_gamma", "exp_a2_shielding", "exp_a6_correctness"]:
            summary_path = RESULTS_DIR / exp_id / "summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text())
                c = s.get("total_cost_usd", 0)
                total_cost += c
                print(f"  {exp_id}: ${c:.4f}")

        print(f"\n  TOTAL API COST: ${total_cost:.4f}")
        print(f"\n  Results saved in: {RESULTS_DIR}/")

        # Save master summary
        master = {
            "experiments_run": list(results.keys()),
            "total_time_s": elapsed,
            "total_cost_usd": total_cost,
        }
        (RESULTS_DIR / "master_summary.json").write_text(json.dumps(master, indent=2))

    print("=" * 70)


if __name__ == "__main__":
    main()
