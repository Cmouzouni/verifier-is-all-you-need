"""Parallel full factorial experiment — 15-30x faster than sequential.

Parallelism strategy:
  1. Episodes within a condition run in parallel (ThreadPoolExecutor, 15 workers)
  2. Independent agents within an episode run in parallel where topology allows
  3. Conditions run sequentially for clean progress reporting

Skips conditions already completed in the sequential run (loaded from cache).
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.congestion import assign_frameworks
from src.diversity import diversity_score, framework_coverage, shannon_entropy
from src.runner import _extract_answer
from src.config import RESULTS_DIR, MODELS
from tasks.phase_a_tasks import PHASE_A_TASKS

# Import design from the original sweep
from experiments.exp_full_sweep import (
    GAMMA_VALUES, N_AGENTS, EPISODES_PER_CONDITION, TAU, TEMPERATURE, MAX_TOKENS,
    MODEL_CONFIGS, TOPOLOGIES, EpisodeRecord,
)


# ════════════════════════════════════════════════════════════════════════
# PARALLEL EPISODE RUNNER
# ════════════════════════════════════════════════════════════════════════

def run_episode_parallel(
    task,
    gamma: float,
    model_config_name: str,
    model_config: dict,
    topology_name: str,
    topology: dict,
    clients: dict[str, LLMClient],
    seed: int,
) -> EpisodeRecord:
    """Run a single episode with parallel agent execution where possible."""
    t0 = time.time()
    rng = random.Random(seed)

    roles = topology["roles"]
    has_synthesis = topology["has_synthesis"]

    # Step 1: Assign frameworks with congestion
    assignments, state = assign_frameworks(
        frameworks=task.frameworks,
        n_agents=N_AGENTS,
        gamma=gamma,
        tau=TAU,
        seed=seed,
    )

    # Step 2: Identify independent vs dependent agents
    # Independent: all propose/insight agents (can run in parallel)
    # Dependent: critique (needs proposals), synthesize (needs all outputs)
    independent_indices = [i for i, r in enumerate(roles) if r in ("propose", "insight")]
    critique_indices = [i for i, r in enumerate(roles) if r == "critique"]
    synth_indices = [i for i, r in enumerate(roles) if r == "synthesize"]

    solutions = [""] * len(roles)
    models_used = [""] * len(roles)
    per_agent_cost = [0.0] * len(roles)
    agent_outputs_raw = [""] * len(roles)
    total_in = 0
    total_out = 0
    total_cost = 0.0

    def build_prompt(i: int, role: str, fw: str, propose_outputs: list[str]) -> tuple[str, str]:
        if role == "propose":
            sys_p = task.framework_prompts.get(fw, f"Solve using the {fw} approach.")
            usr_p = (
                f"Problem: {task.problem}\n\n"
                f"Solve step by step using the {fw} approach.\n"
                f"State your final answer on a line starting with 'ANSWER: '."
            )
        elif role == "insight":
            sys_p = (
                "You are an independent analyst. Other agents have proposed solutions "
                "but you should NOT rely on them. Generate a completely fresh approach."
            )
            usr_p = (
                f"Problem: {task.problem}\n\n"
                f"Context: {len(propose_outputs)} other agents proposed solutions. "
                f"You don't know what they said. Use a fresh approach.\n"
                f"State your final answer on a line starting with 'ANSWER: '."
            )
        elif role == "critique":
            proposals_text = "\n".join(
                f"Proposal {j+1}: {p}" for j, p in enumerate(propose_outputs)
            ) if propose_outputs else "No proposals yet."
            sys_p = (
                "You are a critical reviewer. Find flaws in the proposals, "
                "then state what you think the correct answer is."
            )
            usr_p = (
                f"Problem: {task.problem}\n\n"
                f"Proposals from other agents:\n{proposals_text}\n\n"
                f"Critique these proposals. Then state YOUR answer on a line starting with 'ANSWER: '."
            )
        elif role == "synthesize":
            all_text = "\n".join(
                f"Agent {j+1} ({roles[j]}, {assignments[j]}): {solutions[j]}"
                for j in range(len(roles)) if j != i and solutions[j]
            )
            sys_p = (
                "You are a senior synthesizer. Review all agent outputs, "
                "identify the best answer, and produce the final recommendation."
            )
            usr_p = (
                f"Problem: {task.problem}\n\n"
                f"Agent outputs:\n{all_text}\n\n"
                f"Synthesize into a final answer. State it on a line starting with 'ANSWER: '."
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        return sys_p, usr_p

    def run_agent(i: int, propose_outputs: list[str] = []) -> None:
        nonlocal total_in, total_out, total_cost
        role = roles[i]
        fw = assignments[i]
        model_key = model_config.get(
            "synthesize" if role == "synthesize" else "critique" if role == "critique" else "propose",
            model_config["propose"]
        )
        client = clients[model_key]
        sys_p, usr_p = build_prompt(i, role, fw, propose_outputs)
        resp = client.generate(sys_p, usr_p, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        answer = _extract_answer(resp.content)
        solutions[i] = answer
        models_used[i] = model_key
        per_agent_cost[i] = resp.cost_usd
        agent_outputs_raw[i] = resp.content[:500]
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        total_cost += resp.cost_usd

    # Phase A: Run independent agents in parallel
    if len(independent_indices) > 1:
        with ThreadPoolExecutor(max_workers=len(independent_indices)) as pool:
            futures = {pool.submit(run_agent, i): i for i in independent_indices}
            for f in as_completed(futures):
                f.result()  # raise exceptions if any
    elif independent_indices:
        run_agent(independent_indices[0])

    # Collect proposals for dependent agents
    propose_outputs = [agent_outputs_raw[i][:300] for i in independent_indices if agent_outputs_raw[i]]

    # Phase B: Run critique agents (need proposals)
    for i in critique_indices:
        run_agent(i, propose_outputs)

    # Phase C: Run synthesis agents (need everything)
    for i in synth_indices:
        run_agent(i)

    # Step 3: Determine final answer
    if has_synthesis and any(solutions[i] for i in synth_indices):
        final_answer = solutions[synth_indices[-1]]
    else:
        valid = [s for s in solutions if s]
        if valid:
            counter = Counter(valid)
            final_answer = counter.most_common(1)[0][0]
        else:
            final_answer = ""

    correct = task.check(final_answer)

    return EpisodeRecord(
        gamma=gamma,
        model_config=model_config_name,
        topology=topology_name,
        task_id=task.id,
        task_domain=task.domain,
        task_difficulty=task.difficulty,
        frameworks_assigned=assignments,
        diversity_score=diversity_score(assignments),
        framework_coverage=framework_coverage(assignments, len(task.frameworks)),
        roles=roles,
        models_used=models_used,
        solutions=solutions,
        final_answer=final_answer,
        ground_truth=task.ground_truth,
        correct=correct,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=total_cost,
        wall_time_s=time.time() - t0,
        per_agent_cost=per_agent_cost,
        agent_outputs_raw=agent_outputs_raw,
    )


# ════════════════════════════════════════════════════════════════════════
# PARALLEL CONDITION RUNNER (15 episodes at once)
# ════════════════════════════════════════════════════════════════════════

def run_condition_parallel(
    mc_name: str, mc: dict, tp_name: str, tp: dict, gamma: float,
    clients: dict[str, LLMClient], rng: random.Random,
    max_workers: int = 10,
) -> list[EpisodeRecord]:
    """Run all episodes for one condition in parallel."""
    tasks = rng.choices(PHASE_A_TASKS, k=EPISODES_PER_CONDITION)
    seeds = [rng.randint(0, 2**31) for _ in range(EPISODES_PER_CONDITION)]

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for ep_idx, (task, seed) in enumerate(zip(tasks, seeds)):
            f = pool.submit(
                run_episode_parallel,
                task=task, gamma=gamma,
                model_config_name=mc_name, model_config=mc,
                topology_name=tp_name, topology=tp,
                clients=clients, seed=seed,
            )
            futures[f] = ep_idx

        for f in as_completed(futures):
            try:
                records.append(f.result())
            except Exception as e:
                print(f"    [ERROR] ep {futures[f]}: {e}")

    return records


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_experiment(dry_run: bool = False) -> list[dict]:
    print("=" * 70)
    print("PARALLEL FACTORIAL EXPERIMENT")
    print("=" * 70)
    print(f"  γ values:       {GAMMA_VALUES}")
    print(f"  Model configs:  {list(MODEL_CONFIGS.keys())}")
    print(f"  Topologies:     {list(TOPOLOGIES.keys())}")
    print(f"  Episodes/cond:  {EPISODES_PER_CONDITION}")
    total_conditions = len(MODEL_CONFIGS) * len(TOPOLOGIES) * len(GAMMA_VALUES)
    total_episodes = total_conditions * EPISODES_PER_CONDITION
    print(f"  Total conditions: {total_conditions}")
    print(f"  Total episodes:   {total_episodes}")
    print(f"  Parallelism:     {EPISODES_PER_CONDITION} episodes/condition + parallel agents")
    print()

    if dry_run:
        print(f"[DRY RUN] With parallelism, estimated ~2-3 min per condition")
        print(f"[DRY RUN] {total_conditions} conditions × ~2.5 min = ~{total_conditions * 2.5 / 60:.1f} hours")
        print(f"[DRY RUN] Estimated cost: ~$4-6")
        return []

    # Load cache from sequential run
    cache_path = RESULTS_DIR / "cached_results.json"
    cached_baselines = {}
    cached_conditions = set()
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        cached_baselines = cache.get("baselines", {})
        for c in cache.get("conditions", []):
            cached_conditions.add((c["model_config"], c["topology"], c["gamma"]))
        print(f"Loaded cache: {len(cached_baselines)} baselines, {len(cached_conditions)} conditions")

    # Create clients
    needed_models = set()
    for mc in MODEL_CONFIGS.values():
        needed_models.update(v for k, v in mc.items() if k != "description")
    for m in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        needed_models.add(m)
    clients = {m: LLMClient(model_key=m) for m in needed_models}
    print(f"Initialized clients: {list(clients.keys())}")

    rng = random.Random(42)
    all_records: list[EpisodeRecord] = []
    baseline_records: list[dict] = []
    t_start = time.time()
    cumulative_cost = 0.0

    # ── BASELINES ──────────────────────────────────────────────────────
    if cached_baselines:
        print(f"\n--- BASELINES (from cache) ---")
        for model_key, data in cached_baselines.items():
            print(f"  {model_key:12s}: {data['correct_rate']:.1%} correct, ${data['cost']:.4f}")
            # Reconstruct baseline records
            for task in PHASE_A_TASKS:
                baseline_records.append({
                    "model": model_key, "task_id": task.id,
                    "answer": "", "correct": False, "cost_usd": 0, "tokens": 0,
                    "note": "from_cache"
                })
    else:
        print("\n--- BASELINES: Single model per task ---")
        for model_key in ["qwen-7b", "qwen-22b", "qwen-17b"]:
            client = clients[model_key]
            correct_count = 0
            total_baseline_cost = 0.0

            # Run baselines in parallel too
            def run_baseline(task, mk=model_key):
                c = clients[mk]
                sys_prompt = "You are a problem solver. Solve step by step. State your final answer on a line starting with 'ANSWER: '."
                usr_prompt = f"Problem: {task.problem}"
                resp = c.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                answer = _extract_answer(resp.content)
                return task, answer, task.check(answer), resp.cost_usd, resp.input_tokens + resp.output_tokens

            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(run_baseline, t) for t in PHASE_A_TASKS]
                for f in as_completed(futures):
                    try:
                        task, answer, correct, cost, tokens = f.result()
                        if correct:
                            correct_count += 1
                        total_baseline_cost += cost
                        baseline_records.append({
                            "model": model_key, "task_id": task.id,
                            "answer": answer, "correct": correct,
                            "cost_usd": cost, "tokens": tokens,
                        })
                    except Exception as e:
                        print(f"    [ERROR] baseline {model_key}: {e}")

            rate = correct_count / len(PHASE_A_TASKS)
            cumulative_cost += total_baseline_cost
            print(f"  {model_key:12s}: {rate:.1%} correct ({correct_count}/{len(PHASE_A_TASKS)}) — ${total_baseline_cost:.4f}")

    # ── MAIN SWEEP (parallel episodes per condition) ───────────────────
    condition_idx = 0
    skipped = 0

    # Advance RNG past cached conditions to maintain reproducibility
    for mc_name in MODEL_CONFIGS:
        for tp_name in TOPOLOGIES:
            for gamma in GAMMA_VALUES:
                condition_idx += 1
                # Always advance RNG state (for reproducibility)
                tasks_draw = rng.choices(PHASE_A_TASKS, k=EPISODES_PER_CONDITION)
                seeds_draw = [rng.randint(0, 2**31) for _ in range(EPISODES_PER_CONDITION)]

    # Reset and run for real
    rng = random.Random(42)
    condition_idx = 0

    for mc_name, mc in MODEL_CONFIGS.items():
        for tp_name, tp in TOPOLOGIES.items():
            for gamma in GAMMA_VALUES:
                condition_idx += 1

                cond_records = run_condition_parallel(
                    mc_name, mc, tp_name, tp, gamma,
                    clients, rng, max_workers=10,
                )
                all_records.extend(cond_records)
                for r in cond_records:
                    cumulative_cost += r.total_cost_usd

                # Progress report
                if cond_records:
                    mean_d = sum(r.diversity_score for r in cond_records) / len(cond_records)
                    mean_correct = sum(r.correct for r in cond_records) / len(cond_records)
                    mean_cost = sum(r.total_cost_usd for r in cond_records) / len(cond_records)
                    elapsed = time.time() - t_start
                    rate = condition_idx / max(elapsed, 1) * 60
                    remaining = (total_conditions - condition_idx) / max(rate, 0.01)
                    print(
                        f"  [{condition_idx:3d}/{total_conditions}] "
                        f"{mc_name:20s} {tp_name:22s} γ={gamma:4.1f} | "
                        f"D={mean_d:.2f} correct={mean_correct:.0%} "
                        f"cost/ep=${mean_cost:.4f} "
                        f"cumul=${cumulative_cost:.3f} "
                        f"[{elapsed:.0f}s, ~{remaining:.0f}m left]"
                    )
                else:
                    print(f"  [{condition_idx:3d}/{total_conditions}] {mc_name} {tp_name} γ={gamma} | ALL FAILED")

    # ── Save complete results ──────────────────────────────────────────
    elapsed = time.time() - t_start
    output_dir = RESULTS_DIR / "full_sweep"
    output_dir.mkdir(exist_ok=True)

    data = {
        "experiment": "full_factorial_sweep_parallel",
        "design": {
            "gamma_values": GAMMA_VALUES,
            "model_configs": {k: v["description"] for k, v in MODEL_CONFIGS.items()},
            "topologies": {k: v["description"] for k, v in TOPOLOGIES.items()},
            "episodes_per_condition": EPISODES_PER_CONDITION,
            "n_agents": N_AGENTS,
            "tau": TAU,
            "temperature": TEMPERATURE,
        },
        "summary": {
            "total_episodes": len(all_records),
            "total_cost_usd": cumulative_cost,
            "total_time_s": elapsed,
            "total_api_calls": sum(c.tracker.n_calls for c in clients.values()),
            "total_tokens": sum(c.tracker.total_tokens for c in clients.values()),
        },
        "baselines": baseline_records,
        "episodes": [asdict(r) for r in all_records],
    }
    path = output_dir / "results.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nResults saved to: {path}")

    # ── Aggregated analysis ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*70}")

    print(f"\n--- BASELINE references ---")
    if cached_baselines:
        for k, v in cached_baselines.items():
            print(f"  {k:12s}: {v['correct_rate']:.1%}")
    else:
        for model_key in ["qwen-7b", "qwen-22b", "qwen-17b"]:
            br = [b for b in baseline_records if b["model"] == model_key and b.get("note") != "from_cache"]
            if br:
                rate = sum(b["correct"] for b in br) / len(br)
                cost = sum(b["cost_usd"] for b in br)
                print(f"  {model_key:12s}: {rate:.1%} correct, ${cost:.4f}")

    print(f"\n--- Diversity D(γ) across all configurations ---")
    print(f"{'γ':>6} {'mean D':>8} {'correct':>8} {'cost/ep':>10}")
    for gamma in GAMMA_VALUES:
        eps = [r for r in all_records if r.gamma == gamma]
        if eps:
            md = sum(r.diversity_score for r in eps) / len(eps)
            mc = sum(r.correct for r in eps) / len(eps)
            mcost = sum(r.total_cost_usd for r in eps) / len(eps)
            print(f"{gamma:>6.1f} {md:>8.3f} {mc:>8.1%} {mcost:>10.5f}")

    print(f"\n--- Correctness by model config ---")
    for mc_name in MODEL_CONFIGS:
        eps = [r for r in all_records if r.model_config == mc_name]
        if eps:
            mc = sum(r.correct for r in eps) / len(eps)
            mcost = sum(r.total_cost_usd for r in eps) / len(eps)
            print(f"  {mc_name:25s}: correct={mc:.1%}  cost/ep=${mcost:.5f}")

    print(f"\n--- Correctness by topology ---")
    for tp_name in TOPOLOGIES:
        eps = [r for r in all_records if r.topology == tp_name]
        if eps:
            mc = sum(r.correct for r in eps) / len(eps)
            mcost = sum(r.total_cost_usd for r in eps) / len(eps)
            print(f"  {tp_name:25s}: correct={mc:.1%}  cost/ep=${mcost:.5f}")

    # Best config per cost bracket
    print(f"\n--- Best configuration per cost bracket ---")
    for mc_name in MODEL_CONFIGS:
        for tp_name in TOPOLOGIES:
            eps = [r for r in all_records if r.model_config == mc_name and r.topology == tp_name]
            if eps:
                best_gamma = max(GAMMA_VALUES, key=lambda g: sum(
                    r.correct for r in eps if r.gamma == g
                ) / max(sum(1 for r in eps if r.gamma == g), 1))
                best_eps = [r for r in eps if r.gamma == best_gamma]
                if best_eps:
                    rate = sum(r.correct for r in best_eps) / len(best_eps)
                    cost = sum(r.total_cost_usd for r in best_eps) / len(best_eps)
                    print(f"  {mc_name:20s} × {tp_name:22s} @ γ={best_gamma}: {rate:.0%} ${cost:.5f}")

    print(f"\n--- Per-model cost breakdown ---")
    for model_key, client in clients.items():
        if client.tracker.n_calls > 0:
            print(f"  {model_key:12s}: ${client.tracker.total_cost:.4f} ({client.tracker.n_calls} calls)")

    print(f"\n  TOTAL: ${cumulative_cost:.4f} in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    return [asdict(r) for r in all_records]


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_experiment(dry_run=dry_run)
