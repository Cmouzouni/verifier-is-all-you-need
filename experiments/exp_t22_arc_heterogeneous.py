"""T2.2 (revised) — Heterogeneous game vs single frontier on ARC-AGI.

The right question for ARC: can a SMART game architecture, using cheap models
for breadth and a frontier model where its capability is most useful, MATCH or
BEAT the frontier model used alone — at competitive cost?

The wrong question (which we are not testing): can cheap models alone solve ARC.
We know the answer is no.

Setup:
  - 30 ARC-AGI training problems (sampled with seed 42)
  - 8 conditions: 3 single-model baselines + 5 game architectures

Conditions:
  Single baselines (one model, one shot):
    1. qwen-7b   single
    2. qwen-22b  single
    3. qwen-17b  single (with thinking)

  Heterogeneous game architectures (4 proposers + synthesizer):
    4. all-7b synth-22b      : 4×qwen-7b proposers, qwen-22b synthesizer
    5. all-7b synth-17b      : 4×qwen-7b proposers, qwen-17b synthesizer (frontier)
    6. all-22b synth-17b     : 4×qwen-22b proposers, qwen-17b synthesizer (frontier)
    7. mixed-22b-7b-17b      : 2×qwen-22b + 2×qwen-7b proposers, qwen-17b synthesizer
    8. mixed-7b-17b          : 3×qwen-7b + 1×qwen-22b proposers, qwen-17b synthesizer

Each (condition, problem) gets 2 episodes for noise reduction = 60 trials per
condition, 480 total. Baselines are 1 episode each = 30 trials per baseline,
90 total.

Verification: exact grid match against ground truth.

The decisive comparison: condition (game) vs condition 3 (single frontier).
- If game > frontier alone: heterogeneous architecture provides real value.
- If game ≤ frontier alone: cheap proposers add noise; the frontier model is
  better off solving alone.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, parse_grid_answer


# ── Configuration ──────────────────────────────────────────────────────
N_PROBLEMS = 30
EPISODES_PER_PROBLEM_GAME = 2
EPISODES_PER_PROBLEM_BASELINE = 1
TEMPERATURE = 0.8
MAX_TOKENS = 2048

# Game architecture definitions: list of model_keys for the 4 proposers
GAME_CONFIGS = {
    "all-7b__synth-22b": {
        "proposers": ["qwen-7b", "qwen-7b", "qwen-7b", "qwen-7b"],
        "synthesizer": "qwen-22b",
    },
    "all-7b__synth-17b": {
        "proposers": ["qwen-7b", "qwen-7b", "qwen-7b", "qwen-7b"],
        "synthesizer": "qwen-17b",
    },
    "all-22b__synth-17b": {
        "proposers": ["qwen-22b", "qwen-22b", "qwen-22b", "qwen-22b"],
        "synthesizer": "qwen-17b",
    },
    "mixed-22b-7b__synth-17b": {
        "proposers": ["qwen-22b", "qwen-22b", "qwen-7b", "qwen-7b"],
        "synthesizer": "qwen-17b",
    },
    "mostly-7b__synth-17b": {
        "proposers": ["qwen-7b", "qwen-7b", "qwen-7b", "qwen-22b"],
        "synthesizer": "qwen-17b",
    },
}

BASELINES = ["qwen-7b", "qwen-22b", "qwen-17b"]


def run_baseline_one(client: LLMClient, task) -> dict:
    """One baseline attempt at one ARC task."""
    sys_prompt = (
        "You are a visual-spatial reasoning expert solving ARC-AGI puzzles. "
        "You will be shown several input-output grid pairs that follow the same "
        "transformation rule. Figure out the rule and apply it to a new input grid. "
        "Output the resulting grid in the exact same format."
    )
    usr_prompt = task.format_prompt()
    try:
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        correct = task.check(resp.content)
        return {
            "correct": correct,
            "answer_preview": resp.content[:300],
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        }
    except Exception as e:
        return {"correct": False, "answer_preview": "", "tokens": 0, "cost": 0, "error": str(e)[:80]}


def run_proposer(client: LLMClient, task, framing: str) -> dict:
    """One proposer that tries to solve the ARC task with a particular framing."""
    framings = {
        "color_pattern": "Focus on color patterns and how colors map between input and output.",
        "shape_motion": "Focus on shapes and how they move, rotate, or transform.",
        "counting": "Focus on counting elements and how counts change between input and output.",
        "symmetry": "Focus on symmetry, mirroring, and reflection patterns.",
        "default": "Find the underlying transformation rule.",
    }
    framing_text = framings.get(framing, framings["default"])
    sys_prompt = (
        "You are a visual-spatial reasoning expert solving an ARC-AGI puzzle. "
        f"Analytical strategy: {framing_text} "
        "Reason carefully and output the resulting grid."
    )
    usr_prompt = task.format_prompt()
    try:
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        return {
            "framing": framing,
            "content": resp.content[:2000],
            "answer_correct": task.check(resp.content),
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        }
    except Exception as e:
        return {"framing": framing, "content": "", "answer_correct": False, "tokens": 0, "cost": 0, "error": str(e)[:80]}


def run_synthesizer(client: LLMClient, task, proposals: list) -> dict:
    """The synthesizer reads all proposals and produces the final answer."""
    proposals_text = "\n\n".join(
        f"---\nProposer {i+1} (strategy: {p['framing']}):\n{p['content']}"
        for i, p in enumerate(proposals)
    )
    sys_prompt = (
        "You are a senior ARC-AGI solver. Several other agents have analyzed "
        "this puzzle from different strategic perspectives. Read each agent's "
        "analysis carefully, identify which (if any) found the right transformation "
        "rule, and produce the final correct output grid. You may ignore agents "
        "whose reasoning is wrong, combine insights from multiple agents, or "
        "develop your own answer if none of the proposals are correct."
    )
    usr_prompt = (
        f"{task.format_prompt()}\n\n"
        f"=== PROPOSER ANALYSES ===\n{proposals_text}\n\n"
        f"=== YOUR FINAL ANSWER ==="
    )
    try:
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        return {
            "correct": task.check(resp.content),
            "answer_preview": resp.content[:500],
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        }
    except Exception as e:
        return {"correct": False, "answer_preview": "", "tokens": 0, "cost": 0, "error": str(e)[:80]}


def run_game_episode(clients: dict, task, config: dict) -> dict:
    """One episode of the heterogeneous game on one ARC task."""
    proposer_models = config["proposers"]
    synth_model = config["synthesizer"]
    framings = ["color_pattern", "shape_motion", "counting", "symmetry"]

    # Run all proposers in parallel (each with its own framing)
    with ThreadPoolExecutor(max_workers=len(proposer_models)) as pool:
        futures = []
        for i, m in enumerate(proposer_models):
            futures.append(pool.submit(run_proposer, clients[m], task, framings[i % len(framings)]))
        proposals = [f.result() for f in futures]

    # Run synthesizer
    synth_result = run_synthesizer(clients[synth_model], task, proposals)

    proposal_cost = sum(p["cost"] for p in proposals)
    proposal_tokens = sum(p["tokens"] for p in proposals)

    # How many individual proposers got it right?
    n_proposers_correct = sum(1 for p in proposals if p.get("answer_correct"))

    return {
        "task_id": task.id,
        "n_proposers_correct": n_proposers_correct,
        "synth_correct": synth_result["correct"],
        "proposal_cost": proposal_cost,
        "proposal_tokens": proposal_tokens,
        "synth_cost": synth_result["cost"],
        "synth_tokens": synth_result["tokens"],
        "total_cost": proposal_cost + synth_result["cost"],
        "total_tokens": proposal_tokens + synth_result["tokens"],
    }


def main(dry_run: bool = False):
    print("=" * 70)
    print("T2.2 (revised) — HETEROGENEOUS GAME vs FRONTIER on ARC-AGI")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")
    print(f"  Game configs: {len(GAME_CONFIGS)}")
    print(f"  Episodes per (config, task): {EPISODES_PER_PROBLEM_GAME} (game), {EPISODES_PER_PROBLEM_BASELINE} (baseline)")
    n_baseline_eps = len(BASELINES) * N_PROBLEMS * EPISODES_PER_PROBLEM_BASELINE
    n_game_eps = len(GAME_CONFIGS) * N_PROBLEMS * EPISODES_PER_PROBLEM_GAME
    print(f"  Total episodes: {n_baseline_eps} baselines + {n_game_eps} game = {n_baseline_eps + n_game_eps}")

    if dry_run:
        # Each baseline episode = 1 call. Each game episode = 5 calls (4 prop + 1 synth)
        n_calls_baseline = n_baseline_eps
        n_calls_game = n_game_eps * 5
        # Mix of model costs; assume avg ~$0.0008 per call (mix of 7b/22b/17b, large grids)
        est = (n_calls_baseline + n_calls_game) * 0.0008
        print(f"\n[DRY RUN] {n_calls_baseline + n_calls_game} calls, est ${est:.2f}")
        return

    print(f"\nLoading ARC-AGI...")
    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  {len(tasks)} tasks loaded")

    # Initialize all model clients
    clients = {m: LLMClient(m) for m in ["qwen-7b", "qwen-22b", "qwen-17b"]}

    rng = random.Random(42)
    t_start = time.time()
    output_dir = RESULTS_DIR / "t22_arc_hetero"
    output_dir.mkdir(exist_ok=True)

    all_results = {"baselines": {}, "games": {}}

    # ── Baselines ──────────────────────────────────────────────────────
    print("\n--- BASELINES (single model, 1 attempt each) ---", flush=True)
    for model in BASELINES:
        client = clients[model]
        print(f"  Running {model}...", flush=True)
        records = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(run_baseline_one, client, t) for t in tasks]
            for f in as_completed(futures):
                records.append(f.result())
        k = sum(r["correct"] for r in records)
        n = len(records)
        cost = sum(r["cost"] for r in records)
        all_results["baselines"][model] = {
            "k": k, "n": n, "rate": k / n,
            "total_cost": cost, "cost_per_problem": cost / n,
            "records": records,
        }
        print(f"    {model}: {k}/{n} ({k/n:.1%}) — total ${cost:.4f} (${cost/n:.5f}/problem)", flush=True)

    # Save baselines immediately so we have something even if games fail
    json.dump({
        "phase": "baselines_only",
        "results": all_results,
    }, open(output_dir / "baselines.json", "w"), indent=2, default=str)

    # ── Game configurations ────────────────────────────────────────────
    print("\n--- GAME ARCHITECTURES ---", flush=True)
    for config_name, config in GAME_CONFIGS.items():
        print(f"\n  Game: {config_name}", flush=True)
        print(f"    proposers: {config['proposers']}", flush=True)
        print(f"    synth: {config['synthesizer']}", flush=True)

        # Build job list
        jobs = []
        for task in tasks:
            for ep in range(EPISODES_PER_PROBLEM_GAME):
                jobs.append(task)

        records = []
        # Use small worker pool here because each episode itself uses inner parallelism
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_game_episode, clients, t, config) for t in jobs]
            completed = 0
            for f in as_completed(futures):
                try:
                    records.append(f.result())
                    completed += 1
                    if completed % 10 == 0:
                        k = sum(r["synth_correct"] for r in records)
                        cost = sum(r["total_cost"] for r in records)
                        print(f"    [{completed}/{len(jobs)}] {k}/{completed} correct, cost ${cost:.4f}", flush=True)
                except Exception as e:
                    print(f"    [ERROR] {e}", flush=True)

        if records:
            k = sum(r["synth_correct"] for r in records)
            n = len(records)
            avg_cost = sum(r["total_cost"] for r in records) / n
            avg_proposers_correct = sum(r["n_proposers_correct"] for r in records) / n
            all_results["games"][config_name] = {
                "k": k, "n": n, "rate": k / n,
                "avg_cost_per_episode": avg_cost,
                "avg_proposers_correct": avg_proposers_correct,
                "config": config,
                "records": records,
            }
            print(f"    → {config_name}: {k}/{n} ({k/n:.1%}), avg ${avg_cost:.5f}/ep, "
                  f"avg {avg_proposers_correct:.2f}/4 proposers right", flush=True)

    elapsed = time.time() - t_start
    total_cost = sum(c.tracker.total_cost for c in clients.values())

    # ── Final analysis ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"T2.2 RESULTS — Heterogeneous Game vs Frontier on ARC-AGI")
    print(f"{'='*70}")

    print(f"\n--- BASELINES (single-model, single-shot) ---")
    for model in BASELINES:
        b = all_results["baselines"][model]
        print(f"  {model:12s}: {b['rate']:6.1%} ({b['k']}/{b['n']})  ${b['cost_per_problem']:.5f}/problem")

    frontier = all_results["baselines"]["qwen-17b"]
    frontier_rate = frontier["rate"]
    frontier_cost = frontier["cost_per_problem"]

    print(f"\n--- GAME ARCHITECTURES vs FRONTIER ---")
    for config_name in GAME_CONFIGS:
        if config_name not in all_results["games"]:
            continue
        g = all_results["games"][config_name]
        delta_correct = g["rate"] - frontier_rate
        cost_ratio = g["avg_cost_per_episode"] / frontier_cost if frontier_cost > 0 else float("inf")
        marker = "✓" if delta_correct > 0 else "~" if delta_correct == 0 else "✗"
        print(f"  {marker} {config_name:30s}: {g['rate']:6.1%}  Δ vs frontier: {delta_correct:+5.1%}  "
              f"cost: ${g['avg_cost_per_episode']:.5f}/ep ({cost_ratio:.2f}x frontier)")

    # Decision rule
    print(f"\n--- INTERPRETATION ---")
    best_game = max(all_results["games"].items(), key=lambda x: x[1]["rate"]) if all_results["games"] else None
    if best_game:
        bn, bg = best_game
        if bg["rate"] > frontier_rate:
            cost_ratio = bg["avg_cost_per_episode"] / frontier_cost if frontier_cost > 0 else float("inf")
            print(f"  ✓ BEST GAME ({bn}) BEATS FRONTIER:")
            print(f"     {bg['rate']:.1%} vs frontier {frontier_rate:.1%} (Δ={bg['rate']-frontier_rate:+.1%})")
            print(f"     at {cost_ratio:.2f}x cost (${bg['avg_cost_per_episode']:.5f} vs ${frontier_cost:.5f}/prob)")
            print(f"  → The heterogeneous game architecture provides real value on ARC.")
        elif bg["rate"] == frontier_rate:
            print(f"  ~ BEST GAME ({bn}) MATCHES FRONTIER ({bg['rate']:.1%})")
            print(f"  → No quality improvement; assess cost difference for the recommendation.")
        else:
            print(f"  ✗ BEST GAME ({bn}) UNDERPERFORMS FRONTIER ({bg['rate']:.1%} vs {frontier_rate:.1%})")
            print(f"  → Heterogeneous game does not help on ARC. Cheap proposers add noise.")
            print(f"  → Either pivot benchmark or rethink the architecture.")

    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save (drop the heaviest fields to keep file size reasonable)
    save = {
        "experiment": "t22_arc_heterogeneous",
        "n_problems": len(tasks),
        "baselines": {m: {k: v for k, v in d.items() if k != "records"} for m, d in all_results["baselines"].items()},
        "games": {c: {k: v for k, v in d.items() if k != "records"} for c, d in all_results["games"].items()},
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }
    json.dump(save, open(output_dir / "results.json", "w"), indent=2, default=str)

    # Save raw separately
    json.dump({
        "baselines": {m: d["records"] for m, d in all_results["baselines"].items()},
        "games": {c: d["records"] for c, d in all_results["games"].items()},
    }, open(output_dir / "raw.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
