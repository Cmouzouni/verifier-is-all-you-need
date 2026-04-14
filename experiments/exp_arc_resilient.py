"""Resilient ARC experiment runner — designed to survive Together AI outages.

Features:
  - Persistent state (saved after every completed task to results.json)
  - Resume from disk on restart
  - API health check between batches; sleeps when API is down
  - Per-task retries with exponential backoff
  - Priority-ordered experiment queue
  - Aggressive parallelism when API is healthy

Designed to be launched once and left running unattended overnight if needed.

Usage:
  python -m experiments.exp_arc_resilient                    # run all phases
  python -m experiments.exp_arc_resilient --resume           # resume from saved state
  python -m experiments.exp_arc_resilient --only baseline_17b  # one phase only
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, parse_grid_answer, ARCTask
from experiments.exp_arc_crack import (
    call_arc, synthesize_over_candidates, FRAMINGS,
    parse_canonical, check_canonical,
)


# ── Configuration ──────────────────────────────────────────────────────
N_PROBLEMS = 30
WORKERS = 8  # conservative — let's not stress the API
MAX_TASK_RETRIES = 5  # per-task retry limit
HEALTH_CHECK_TIMEOUT = 30
HEALTH_CHECK_BACKOFF_S = 60  # wait this long when API is unhealthy


# ── Persistent state ───────────────────────────────────────────────────
STATE_PATH = RESULTS_DIR / "arc_resilient" / "state.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "started": time.time(),
        "phases": {},  # phase_name → results
        "completed_phases": [],
        "elapsed_s": 0.0,
        "total_cost_usd": 0.0,
    }


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ── API health check ───────────────────────────────────────────────────
def check_api_health(model_keys: list[str], clients: dict) -> dict[str, bool]:
    """Test each required model with a tiny call. Returns {model: ok_bool}."""
    health = {}
    for m in model_keys:
        client = clients[m]
        try:
            t0 = time.time()
            resp = client.generate(
                "Answer concisely.",
                "What is 2+2?",
                temperature=0.1, max_tokens=32,
                thinking=(m == "qwen-17b"),
                _max_retries=1,  # don't burn retries on health check
            )
            health[m] = True
        except Exception as e:
            health[m] = False
    return health


def wait_for_api_health(required: list[str], clients: dict, max_wait_s: int = 3600) -> bool:
    """Wait until all required models are healthy. Returns True if recovered."""
    waited = 0
    while waited < max_wait_s:
        health = check_api_health(required, clients)
        unhealthy = [m for m, ok in health.items() if not ok]
        if not unhealthy:
            return True
        print(f"  [HEALTH CHECK] unhealthy: {unhealthy}; waiting {HEALTH_CHECK_BACKOFF_S}s "
              f"(total wait: {waited}s)", flush=True)
        time.sleep(HEALTH_CHECK_BACKOFF_S)
        waited += HEALTH_CHECK_BACKOFF_S
    return False


# ── Per-task wrappers with retries ─────────────────────────────────────
def call_arc_resilient(client: LLMClient, task, framing: str = None,
                        max_attempts: int = MAX_TASK_RETRIES) -> dict:
    """Wrapper around call_arc with extra resilience."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            r = call_arc(client, task, framing=framing)
            if "error" not in r:
                return r
            last_err = r.get("error", "unknown")
        except Exception as e:
            last_err = str(e)[:80]
        # Backoff
        time.sleep(2 ** attempt)
    return {"correct": False, "content": "", "tokens": 0, "cost": 0, "error": f"max_retries: {last_err}"}


# ── Phases ─────────────────────────────────────────────────────────────

def phase_baseline(name: str, model_key: str, tasks: list[ARCTask], clients: dict) -> dict:
    """Single-shot baseline for one model."""
    print(f"\n[PHASE] BASELINE: {name}", flush=True)
    client = clients[model_key]
    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(call_arc_resilient, client, t): t.id for t in tasks}
        for f in as_completed(futures):
            try:
                records.append(f.result())
            except Exception as e:
                records.append({"correct": False, "cost": 0, "error": str(e)[:80]})
    k = sum(r["correct"] for r in records)
    n = len(records)
    cost = sum(r.get("cost", 0) for r in records)
    print(f"  → {name}: {k}/{n} ({k/n:.1%}) cost ${cost:.4f}", flush=True)
    return {"name": name, "type": "baseline", "k": k, "n": n, "rate": k / n, "cost": cost}


def phase_best_of_n(name: str, model_key: str, tasks: list[ARCTask], clients: dict, n_samples: int) -> dict:
    """Best-of-N (oracle) — count problems where any of N samples is correct."""
    print(f"\n[PHASE] BEST-OF-{n_samples}: {name}", flush=True)
    client = clients[model_key]
    results_by_task = defaultdict(list)
    jobs = [(t, FRAMINGS[i % len(FRAMINGS)]) for t in tasks for i in range(n_samples)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_task = {pool.submit(call_arc_resilient, client, t, fr): t.id for t, fr in jobs}
        for f in as_completed(future_to_task):
            tid = future_to_task[f]
            try:
                results_by_task[tid].append(f.result())
            except Exception as e:
                results_by_task[tid].append({"correct": False, "cost": 0, "error": str(e)[:80]})
    per_task_correct = []
    total_cost = 0
    for t in tasks:
        recs = results_by_task[t.id]
        per_task_correct.append(any(r["correct"] for r in recs))
        total_cost += sum(r.get("cost", 0) for r in recs)
    k = sum(per_task_correct)
    print(f"  → {name} best-of-{n_samples}: {k}/{len(tasks)} ({k/len(tasks):.1%}) cost ${total_cost:.4f}", flush=True)
    return {
        "name": f"{name}_best_of_{n_samples}", "type": "oracle",
        "k": k, "n": len(tasks), "rate": k / len(tasks), "cost": total_cost,
    }


def phase_synth_game(name: str, prop_models: list[str], synth_model: str,
                      tasks: list[ARCTask], clients: dict) -> dict:
    """N proposers (one per prop_models entry) + 1 synthesizer."""
    n_props = len(prop_models)
    print(f"\n[PHASE] SYNTH GAME: {name} ({n_props} proposers, {synth_model} synth)", flush=True)

    def run_one(task):
        # Run proposers in parallel
        with ThreadPoolExecutor(max_workers=n_props) as inner:
            futures = [
                inner.submit(call_arc_resilient, clients[m], task, FRAMINGS[i % len(FRAMINGS)])
                for i, m in enumerate(prop_models)
            ]
            proposals = [f.result() for f in futures]
        # Synthesize
        try:
            synth = synthesize_over_candidates(clients[synth_model], task, proposals)
        except Exception as e:
            synth = {"correct": False, "cost": 0, "error": str(e)[:80]}
        return {
            "task_id": task.id,
            "n_proposers_correct": sum(1 for p in proposals if p["correct"]),
            "synth_correct": synth["correct"],
            "total_cost": sum(p.get("cost", 0) for p in proposals) + synth.get("cost", 0),
        }

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        for f in as_completed(futures):
            try:
                records.append(f.result())
            except Exception as e:
                records.append({"task_id": "?", "synth_correct": False, "n_proposers_correct": 0, "total_cost": 0, "error": str(e)[:80]})

    k = sum(r["synth_correct"] for r in records)
    n = len(records)
    any_k = sum(1 for r in records if r["n_proposers_correct"] > 0)
    cost = sum(r["total_cost"] for r in records)
    print(f"  → {name}: synth={k}/{n} ({k/n:.1%}), any-prop={any_k}/{n} ({any_k/n:.1%}), cost ${cost:.4f}", flush=True)
    return {
        "name": name, "type": "synth", "k": k, "n": n, "rate": k / n,
        "any_proposer_rate": any_k / n, "cost": cost,
    }


# ── Phase queue ────────────────────────────────────────────────────────
# Each phase is (name, runner_function, args, required_models)
def make_phase_queue(tasks: list[ARCTask], clients: dict):
    return [
        # PRIORITY 1: The fair frontier baseline (most important missing data)
        ("baseline_qwen17b_thinking", phase_baseline, ("qwen-17b w/ thinking", "qwen-17b", tasks, clients), ["qwen-17b"]),

        # PRIORITY 2: Alternative frontier baselines (no qwen-22b dependence)
        ("baseline_deepseek", phase_baseline, ("deepseek-v3.1", "deepseek", tasks, clients), ["deepseek"]),
        ("baseline_llama70b", phase_baseline, ("llama-3.3-70b", "llama-70b", tasks, clients), ["llama-70b"]),

        # PRIORITY 3: Best-of-N with the strongest models
        ("bestofn_qwen17b_8", phase_best_of_n, ("qwen-17b", "qwen-17b", tasks, clients, 8), ["qwen-17b"]),
        ("bestofn_deepseek_8", phase_best_of_n, ("deepseek", "deepseek", tasks, clients, 8), ["deepseek"]),

        # PRIORITY 4: Frontier-only games (the heterogeneous-with-strong-proposers bet)
        ("game_4xds_synthds",
            phase_synth_game,
            ("4xDS_synthDS", ["deepseek"]*4, "deepseek", tasks, clients),
            ["deepseek"]),
        ("game_4x17b_synth17b",
            phase_synth_game,
            ("4x17b_synth17b", ["qwen-17b"]*4, "qwen-17b", tasks, clients),
            ["qwen-17b"]),
        ("game_2x17b_2xds_synth17b",
            phase_synth_game,
            ("2x17b_2xDS_synth17b", ["qwen-17b"]*2 + ["deepseek"]*2, "qwen-17b", tasks, clients),
            ["qwen-17b", "deepseek"]),

        # PRIORITY 5: qwen-22b experiments (will be skipped if 22b is down)
        ("baseline_qwen22b", phase_baseline, ("qwen-22b", "qwen-22b", tasks, clients), ["qwen-22b"]),
        ("bestofn_qwen22b_16", phase_best_of_n, ("qwen-22b", "qwen-22b", tasks, clients, 16), ["qwen-22b"]),
        ("game_8x22b_synth17b",
            phase_synth_game,
            ("8x22b_synth17b", ["qwen-22b"]*8, "qwen-17b", tasks, clients),
            ["qwen-22b", "qwen-17b"]),

        # PRIORITY 6: Cheap baselines (always last; we already have these)
        ("baseline_qwen7b", phase_baseline, ("qwen-7b", "qwen-7b", tasks, clients), ["qwen-7b"]),
    ]


def main():
    print("=" * 70)
    print("ARC RESILIENT RUNNER")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")
    print(f"  Workers: {WORKERS}")
    print(f"  State: {STATE_PATH}")

    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only = sys.argv[idx + 1].split(",")

    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  Loaded {len(tasks)} ARC tasks")

    clients = {
        "qwen-7b": LLMClient("qwen-7b"),
        "qwen-22b": LLMClient("qwen-22b"),
        "qwen-17b": LLMClient("qwen-17b"),
        "deepseek": LLMClient("deepseek"),
        "llama-70b": LLMClient("llama-70b"),
    }

    state = load_state()
    print(f"\nResuming with {len(state['completed_phases'])} phases completed")

    queue = make_phase_queue(tasks, clients)

    # Filter by --only if specified
    if only:
        queue = [(n, fn, args, req) for n, fn, args, req in queue if n in only]
        print(f"  Filtering to {len(queue)} phases: {[n for n, _, _, _ in queue]}")

    failed_phases = []

    for phase_name, fn, args, required in queue:
        if phase_name in state["completed_phases"]:
            print(f"\n[SKIP] {phase_name} (already done)")
            continue

        # Health check
        print(f"\n[HEALTH CHECK] for {phase_name} (requires: {required})", flush=True)
        health = check_api_health(required, clients)
        unhealthy = [m for m, ok in health.items() if not ok]
        if unhealthy:
            print(f"  Unhealthy: {unhealthy}. Waiting up to 10 min...", flush=True)
            recovered = wait_for_api_health(required, clients, max_wait_s=600)
            if not recovered:
                print(f"  [SKIP] {phase_name} — API still down after 10 min", flush=True)
                failed_phases.append(phase_name)
                state["phases"][phase_name] = {"skipped": True, "reason": "api_outage"}
                save_state(state)
                continue

        # Run phase
        try:
            t_phase = time.time()
            result = fn(*args)
            phase_elapsed = time.time() - t_phase
            result["elapsed_s"] = phase_elapsed
            state["phases"][phase_name] = result
            state["completed_phases"].append(phase_name)
            state["total_cost_usd"] = sum(c.tracker.total_cost for c in clients.values())
            save_state(state)
            print(f"  [SAVED] {phase_name} done in {phase_elapsed:.0f}s, cumulative cost ${state['total_cost_usd']:.4f}", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERROR] {phase_name} failed: {e}\n{tb[:500]}", flush=True)
            failed_phases.append(phase_name)
            state["phases"][phase_name] = {"failed": True, "error": str(e)[:200]}
            save_state(state)

    # Retry failed phases once after all primary phases done
    if failed_phases:
        print(f"\n[RETRY] {len(failed_phases)} phases failed: {failed_phases}", flush=True)
        time.sleep(60)
        for phase_name in failed_phases:
            ph = next(((n, fn, args, req) for n, fn, args, req in queue if n == phase_name), None)
            if ph is None:
                continue
            n, fn, args, req = ph
            print(f"\n[RETRY] {n}", flush=True)
            health = check_api_health(req, clients)
            if any(not v for v in health.values()):
                print(f"  Still unhealthy: {[m for m,v in health.items() if not v]}, skipping retry", flush=True)
                continue
            try:
                result = fn(*args)
                state["phases"][n] = result
                if n not in state["completed_phases"]:
                    state["completed_phases"].append(n)
                save_state(state)
            except Exception as e:
                print(f"  [RETRY-ERROR] {n}: {e}", flush=True)
                state["phases"][n] = {"failed": True, "error": str(e)[:200]}
                save_state(state)

    # Final summary
    elapsed = time.time() - state["started"]
    state["elapsed_s"] = elapsed
    state["total_cost_usd"] = sum(c.tracker.total_cost for c in clients.values())
    save_state(state)

    print(f"\n{'='*70}")
    print(f"ARC RESILIENT RUNNER — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\n  Completed: {len(state['completed_phases'])}/{len(queue)} phases")
    print(f"  Total cost: ${state['total_cost_usd']:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print all results sorted by rate
    completed = [(n, r) for n, r in state["phases"].items() if isinstance(r, dict) and "rate" in r]
    completed.sort(key=lambda x: -x[1]["rate"])
    print(f"\n  {'Phase':40s} {'Type':10s} {'Rate':>8s} {'Cost':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*10}")
    for name, r in completed:
        marker = "★" if r["rate"] >= 0.30 else " "
        print(f"  {marker} {name:38s} {r['type']:10s} {r['rate']:>7.1%} ${r['cost']:>8.4f}")


if __name__ == "__main__":
    main()
