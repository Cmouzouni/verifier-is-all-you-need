"""ARC 22B Attack — focused assault using qwen-22b (the strongest single model on ARC).

Findings that drove this design:
  - qwen-22b alone: 30% on ARC-30 (the strongest single-model baseline)
  - qwen-17b with thinking: 6.7% (worse despite being "frontier")
  - qwen-7b alone: 10%
  - Active param count > total: qwen-22b (22B active) >> qwen-17b (17B active)

Goal: push qwen-22b from 30% to 40%+ using:
  1. Best-of-N qwen-22b with framing diversity
  2. Synthesis-completed games with qwen-22b as both proposer and synthesizer
  3. Cross-architecture diversity (qwen-22b + llama-70b proposers)

All experiments use qwen-22b as the synthesizer when possible (it's the best ARC model).

Architectures:
  EXP-1: best-of-8 qwen-22b (oracle) — theoretical ceiling for parallelism
  EXP-2: best-of-16 qwen-22b (oracle, more parallelism)
  EXP-3: 8× qwen-22b proposers + qwen-22b synth (homogeneous high-cap game)
  EXP-4: 4× qwen-22b + 4× llama-70b proposers + qwen-22b synth (mixed-architecture)
  EXP-5: 4× qwen-22b proposers + qwen-22b synth (smaller game, faster)
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
WORKERS = 4  # reduced from 16 to be gentler on the flaky qwen-22b API
STATE_PATH = RESULTS_DIR / "arc_22b_attack" / "state.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"phases": {}, "completed": [], "started": time.time()}


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ── Phases ─────────────────────────────────────────────────────────────

def phase_baseline(name: str, model_key: str, tasks, clients) -> dict:
    print(f"\n[PHASE] BASELINE: {name}", flush=True)
    client = clients[model_key]
    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(call_arc, client, t) for t in tasks]
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


def phase_best_of_n(name: str, model_key: str, tasks, clients, n_samples: int) -> dict:
    print(f"\n[PHASE] BEST-OF-{n_samples}: {name}", flush=True)
    client = clients[model_key]
    results_by_task = defaultdict(list)
    jobs = [(t, FRAMINGS[i % len(FRAMINGS)]) for t in tasks for i in range(n_samples)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_task = {pool.submit(call_arc, client, t, framing=fr): t.id for t, fr in jobs}
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
        "name": f"{name}_best_of_{n_samples}",
        "type": "oracle",
        "k": k, "n": len(tasks), "rate": k / len(tasks), "cost": total_cost,
    }


def phase_majority_vote(name: str, model_key: str, tasks, clients, n_samples: int) -> dict:
    """Consensus by canonical grid match — no synth, just majority vote."""
    print(f"\n[PHASE] MAJORITY VOTE N={n_samples}: {name}", flush=True)
    client = clients[model_key]
    results_by_task = defaultdict(list)
    jobs = [(t, FRAMINGS[i % len(FRAMINGS)]) for t in tasks for i in range(n_samples)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_task = {pool.submit(call_arc, client, t, framing=fr): t.id for t, fr in jobs}
        completed = 0
        for f in as_completed(future_to_task):
            tid = future_to_task[f]
            try:
                results_by_task[tid].append(f.result())
            except Exception as e:
                results_by_task[tid].append({"correct": False, "cost": 0, "error": str(e)[:80]})
            completed += 1
            if completed % 20 == 0:
                tasks_with_any = sum(1 for tid, samples in results_by_task.items() if any(r.get("correct") for r in samples))
                print(f"    [{completed}/{len(jobs)}] {tasks_with_any} tasks have ≥1 correct sample so far", flush=True)

    per_task_correct = []
    any_correct_count = 0
    total_cost = 0
    for t in tasks:
        recs = results_by_task[t.id]
        # Get canonical grids for each
        canonicals = [parse_canonical(r.get("content", "")) for r in recs]
        votes = Counter(c for c in canonicals if c)
        if votes:
            top, _ = votes.most_common(1)[0]
            consensus_correct = check_canonical(top, t)
        else:
            consensus_correct = False
        per_task_correct.append(consensus_correct)
        if any(r["correct"] for r in recs):
            any_correct_count += 1
        total_cost += sum(r.get("cost", 0) for r in recs)

    k = sum(per_task_correct)
    print(f"  → {name} majority-vote N={n_samples}: {k}/{len(tasks)} ({k/len(tasks):.1%}) "
          f"any-correct={any_correct_count}/{len(tasks)} cost ${total_cost:.4f}", flush=True)
    return {
        "name": f"{name}_majvote_{n_samples}",
        "type": "consensus",
        "k": k, "n": len(tasks), "rate": k / len(tasks),
        "any_correct": any_correct_count,
        "cost": total_cost,
    }


def phase_synth_game(name: str, prop_models: list, synth_model: str,
                      tasks, clients) -> dict:
    n_props = len(prop_models)
    print(f"\n[PHASE] SYNTH GAME: {name} ({n_props} proposers, {synth_model} synth)", flush=True)

    def run_one(task):
        with ThreadPoolExecutor(max_workers=n_props) as inner:
            futures = [
                inner.submit(call_arc, clients[m], task, FRAMINGS[i % len(FRAMINGS)])
                for i, m in enumerate(prop_models)
            ]
            proposals = [f.result() for f in futures]
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
        completed = 0
        for f in as_completed(futures):
            try:
                records.append(f.result())
                completed += 1
                if completed % 2 == 0:
                    k = sum(r["synth_correct"] for r in records)
                    any_k = sum(1 for r in records if r["n_proposers_correct"] > 0)
                    cost = sum(r["total_cost"] for r in records)
                    print(f"  [{completed}/{len(tasks)}] synth={k}/{completed} any-prop={any_k}/{completed} cost ${cost:.4f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)

    k = sum(r["synth_correct"] for r in records)
    n = len(records)
    any_k = sum(1 for r in records if r["n_proposers_correct"] > 0)
    cost = sum(r["total_cost"] for r in records)
    print(f"  → {name}: synth={k}/{n} ({k/n:.1%}), any-prop={any_k}/{n} ({any_k/n:.1%}), cost ${cost:.4f}", flush=True)
    return {
        "name": name, "type": "synth", "k": k, "n": n, "rate": k / n,
        "any_proposer_rate": any_k / n, "cost": cost,
    }


def make_queue(tasks, clients):
    return [
        # PRIORITY 1: best-of-N qwen-22b — pure parallelism, theoretical ceiling (DONE = 40%)
        ("bestofn_22b_8", phase_best_of_n, ("qwen-22b", "qwen-22b", tasks, clients, 8)),
        # PRIORITY 2: 4-proposer synth-completed game (small, 150 calls — fits in recovery window)
        ("game_4x22b_synth22b", phase_synth_game, ("4x22b_synth22b", ["qwen-22b"]*4, "qwen-22b", tasks, clients)),
        # PRIORITY 3: majority vote consensus (240 calls)
        ("majvote_22b_8", phase_majority_vote, ("qwen-22b", "qwen-22b", tasks, clients, 8)),
        # PRIORITY 4: 8-proposer synth-completed game (270 calls)
        ("game_8x22b_synth22b", phase_synth_game, ("8x22b_synth22b", ["qwen-22b"]*8, "qwen-22b", tasks, clients)),
        # PRIORITY 5: best-of-16 (more parallelism, 480 calls)
        ("bestofn_22b_16", phase_best_of_n, ("qwen-22b", "qwen-22b", tasks, clients, 16)),
        # PRIORITY 6: cross-architecture (22b + llama)
        ("game_4x22b_4xllama_synth22b", phase_synth_game,
            ("4x22b_4xllama_synth22b", ["qwen-22b"]*4 + ["llama-70b"]*4, "qwen-22b", tasks, clients)),
    ]


def main():
    print("=" * 70)
    print("ARC 22B ATTACK — focused on qwen-22b (the strongest ARC model)")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")
    print(f"  Workers: {WORKERS}")
    print(f"  Goal: push from 30% baseline to 40%+ on ARC")

    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  Loaded {len(tasks)} ARC tasks")

    clients = {
        "qwen-22b": LLMClient("qwen-22b"),
        "llama-70b": LLMClient("llama-70b"),
    }

    state = load_state()
    print(f"\nResuming with {len(state['completed'])} phases completed")

    queue = make_queue(tasks, clients)

    for phase_name, fn, args in queue:
        if phase_name in state["completed"]:
            print(f"\n[SKIP] {phase_name} (already done)")
            continue
        try:
            t_phase = time.time()
            result = fn(*args)
            result["elapsed_s"] = time.time() - t_phase
            state["phases"][phase_name] = result
            state["completed"].append(phase_name)
            save_state(state)
            print(f"  [SAVED] {phase_name} done in {result['elapsed_s']:.0f}s", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERROR] {phase_name} failed: {e}\n{tb[:300]}", flush=True)
            state["phases"][phase_name] = {"failed": True, "error": str(e)[:200]}
            save_state(state)

    # Summary
    elapsed = time.time() - state["started"]
    total_cost = sum(c.tracker.total_cost for c in clients.values())
    state["elapsed_s"] = elapsed
    state["total_cost_usd"] = total_cost
    save_state(state)

    print(f"\n{'='*70}")
    print(f"ARC 22B ATTACK — FINAL")
    print(f"{'='*70}")
    print(f"\n  qwen-22b BASELINE: 30.0% (reference)")
    print(f"\n  {'Phase':40s} {'Type':12s} {'Rate':>8s} {'Cost':>10s}")
    print(f"  {'-'*40} {'-'*12} {'-'*8} {'-'*10}")
    completed = [(n, r) for n, r in state["phases"].items() if isinstance(r, dict) and "rate" in r]
    completed.sort(key=lambda x: -x[1]["rate"])
    for name, r in completed:
        marker = "★" if r["rate"] >= 0.40 else ("●" if r["rate"] >= 0.30 else " ")
        print(f"  {marker} {name:38s} {r['type']:12s} {r['rate']:>7.1%} ${r['cost']:>8.4f}")

    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
