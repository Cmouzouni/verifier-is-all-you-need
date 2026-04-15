"""R5: Bootstrap CIs + hierarchical variance decomposition for the topology comparison.

Addresses reviewer concerns R1#2 and R2#3:
- Wilson CIs assume episode-level independence, but task-level variance may dominate
- No correction for multiple comparisons
- Need to separate task variance from run variance

This script:
1. Computes task-level bootstrap 95% CIs (1000 resamples of 30 tasks)
2. Decomposes variance into task vs run components
3. Applies Bonferroni correction to pairwise topology comparisons
4. Reports whether the topology-dominance finding survives

All from cached data — $0 cost.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

np.random.seed(42)

def load_episodes():
    data = json.load(open("results/full_sweep/results.json"))
    return data["episodes"]


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Task-level bootstrap CI: resample TASKS, not episodes."""
    arr = np.array(values)
    boot_means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lo), float(hi)


def main():
    episodes = load_episodes()

    # ── 1. Task-level rates per topology ──────────────────────────────
    # For each (topology, task), compute the mean correctness across all
    # episodes (model configs × gammas). This gives one rate per task per
    # topology, and the bootstrap resamples tasks.
    by_topo_task = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        by_topo_task[e["topology"]][e["task_id"]].append(1 if e["correct"] else 0)

    SYNTH_TOPOS = {"all-insight-synth", "full-debate", "propose-critique-synth"}
    NOSYNTH_TOPOS = {"propose-only", "propose-critique", "consensus-insight"}

    print("=" * 70)
    print("R5: BOOTSTRAP CIs (task-level, 10K resamples, 30 tasks)")
    print("=" * 70)

    topo_stats = {}
    for topo in sorted(by_topo_task.keys()):
        task_means = [np.mean(v) for v in by_topo_task[topo].values()]
        mean, lo, hi = bootstrap_ci(task_means)
        group = "synth" if topo in SYNTH_TOPOS else "no-synth"
        topo_stats[topo] = {"mean": mean, "lo": lo, "hi": hi, "group": group, "task_means": task_means}
        print(f"  {topo:25s} [{group:8s}]: {mean:.1%}  bootstrap 95% CI [{lo:.1%}, {hi:.1%}]")

    # ── 2. Synth vs no-synth aggregate ────────────────────────────────
    print()
    print("Synth-completed vs synthesis-free (task-level bootstrap):")
    synth_task_means = []
    nosynth_task_means = []
    tasks = sorted(set(e["task_id"] for e in episodes))
    for task in tasks:
        synth_eps = [1 if e["correct"] else 0 for e in episodes
                     if e["task_id"] == task and e["topology"] in SYNTH_TOPOS]
        nosynth_eps = [1 if e["correct"] else 0 for e in episodes
                       if e["task_id"] == task and e["topology"] in NOSYNTH_TOPOS]
        if synth_eps:
            synth_task_means.append(np.mean(synth_eps))
        if nosynth_eps:
            nosynth_task_means.append(np.mean(nosynth_eps))

    s_mean, s_lo, s_hi = bootstrap_ci(synth_task_means)
    n_mean, n_lo, n_hi = bootstrap_ci(nosynth_task_means)
    print(f"  Synth-completed:  {s_mean:.1%}  [{s_lo:.1%}, {s_hi:.1%}]")
    print(f"  Synthesis-free:   {n_mean:.1%}  [{n_lo:.1%}, {n_hi:.1%}]")
    print(f"  Gap: {(s_mean - n_mean)*100:.1f}pp")
    if s_lo > n_hi:
        print(f"  ✅ CIs DO NOT OVERLAP — topology dominance survives task-level bootstrap")
    else:
        print(f"  ⚠️  CIs overlap — topology dominance is weaker under task-level bootstrap")

    # ── 3. Bootstrap test for each pair of topologies ─────────────────
    print()
    print("Pairwise bootstrap tests (Bonferroni-corrected, 15 comparisons, α=0.05/15=0.0033):")
    topo_names = sorted(topo_stats.keys())
    n_comparisons = len(topo_names) * (len(topo_names) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"  Bonferroni α = {bonferroni_alpha:.4f} (corrected for {n_comparisons} comparisons)")
    print()

    for i, t1 in enumerate(topo_names):
        for t2 in topo_names[i+1:]:
            m1 = np.array(topo_stats[t1]["task_means"])
            m2 = np.array(topo_stats[t2]["task_means"])
            # Bootstrap the difference
            diffs = []
            for _ in range(10000):
                idx = np.random.choice(len(m1), size=len(m1), replace=True)
                diffs.append(np.mean(m1[idx]) - np.mean(m2[idx]))
            diffs = np.array(diffs)
            lo = np.percentile(diffs, 100 * bonferroni_alpha / 2)
            hi = np.percentile(diffs, 100 * (1 - bonferroni_alpha / 2))
            mean_diff = np.mean(diffs)
            sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
            print(f"  {t1:25s} vs {t2:25s}: Δ={mean_diff:+.1%}  [{lo:+.1%}, {hi:+.1%}]  {sig}")

    # ── 4. Variance decomposition ─────────────────────────────────────
    print()
    print("=" * 70)
    print("VARIANCE DECOMPOSITION (task vs run)")
    print("=" * 70)
    # For the best topology (all-insight-synth), decompose variance
    topo = "all-insight-synth"
    task_data = by_topo_task[topo]
    task_means = np.array([np.mean(v) for v in task_data.values()])
    task_var = np.var(task_means)  # variance of task-level means

    # Within-task variance (averaged across tasks)
    within_vars = [np.var(v) for v in task_data.values() if len(v) > 1]
    within_var = np.mean(within_vars)

    total_var = task_var + within_var
    print(f"  Topology: {topo}")
    print(f"  Task-level variance:   {task_var:.4f}  ({task_var/total_var:.0%} of total)")
    print(f"  Within-task variance:  {within_var:.4f}  ({within_var/total_var:.0%} of total)")
    print(f"  Total variance:        {total_var:.4f}")
    if task_var > within_var:
        print(f"  ⚠️  Task variance dominates — episode-level Wilson CIs understate uncertainty")
    else:
        print(f"  ✅ Within-task variance dominates — episode-level CIs are approximately valid")


if __name__ == "__main__":
    main()
