"""Full factorial experiment: γ × model_size × topology × role_composition.

This is the comprehensive experiment that generates the complete dataset.
All Qwen models. Full cost tracking. Ground-truth correctness verification.

Dimensions:
  1. γ values: {0, 0.5, 1, 2, 4, 8} — congestion strength
  2. Model configs: {all-7B, all-22B, mixed-7B+22B, mixed-7B+17b} — size as parameter
  3. Topologies: {propose-only, debate, consensus-insight} — graph structure
  4. Role compositions: {homogeneous, heterogeneous} — who plays what role

Primary metrics per episode:
  - diversity_score: D = 1 - max(µ_i)
  - correctness: ground truth verification
  - total_cost_usd: API cost
  - tokens_used: input + output
  - wall_time: latency

Output: one big JSON with every episode, plus per-condition aggregates.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
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


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL DESIGN
# ════════════════════════════════════════════════════════════════════════

GAMMA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]  # core sweep
N_AGENTS = 4
EPISODES_PER_CONDITION = 15  # 15 episodes per condition — good statistical power
TAU = 0.3
TEMPERATURE = 0.7
MAX_TOKENS = 1024  # frontier models need more tokens for step-by-step + answer

# Model configurations: who plays what role
# Each config defines which model to use for propose, critique, and synthesize
MODEL_CONFIGS = {
    # Homogeneous configs — test effect of model size alone
    "all-7b": {
        "propose": "qwen-7b", "critique": "qwen-7b", "synthesize": "qwen-7b",
        "description": "All agents use 7B (cheapest tier)",
    },
    "all-22b": {
        "propose": "qwen-22b", "critique": "qwen-22b", "synthesize": "qwen-22b",
        "description": "All agents use 22B-active MoE (medium tier)",
    },
    # Heterogeneous configs — test role-size matching
    "propose22b-crit7b": {
        "propose": "qwen-22b", "critique": "qwen-7b", "synthesize": "qwen-22b",
        "description": "22B proposes, 7B critiques, 22B synthesizes",
    },
    "propose7b-synth22b": {
        "propose": "qwen-7b", "critique": "qwen-7b", "synthesize": "qwen-22b",
        "description": "7B proposes+critiques, 22B synthesizes only",
    },
    "propose17b-crit7b": {
        "propose": "qwen-17b", "critique": "qwen-7b", "synthesize": "qwen-17b",
        "description": "Frontier 397B proposes+synthesizes, 7B critiques (heterogeneous)",
    },
}

# Topology definitions — different graph structures
# Each topology defines how agents interact
TOPOLOGIES = {
    "propose-only": {
        "description": "Each agent proposes independently, majority vote at end",
        "roles": ["propose"] * 4,
        "has_synthesis": False,
    },
    "propose-critique": {
        "description": "2 agents propose, 2 critique, then majority vote",
        "roles": ["propose", "propose", "critique", "critique"],
        "has_synthesis": False,
    },
    "propose-critique-synth": {
        "description": "2 propose, 1 critiques, 1 synthesizes all (Debate Tree)",
        "roles": ["propose", "propose", "critique", "synthesize"],
        "has_synthesis": True,
    },
    "consensus-insight": {
        "description": "3 consensus (see each other), 1 insight (shielded, fresh approach)",
        "roles": ["propose", "propose", "propose", "insight"],
        "has_synthesis": False,
    },
    "full-debate": {
        "description": "1 propose, 1 critique, 1 insight (shielded), 1 synthesize (full pipeline)",
        "roles": ["propose", "critique", "insight", "synthesize"],
        "has_synthesis": True,
    },
    "all-insight-synth": {
        "description": "3 insight agents (all shielded, max diversity), 1 synthesize",
        "roles": ["insight", "insight", "insight", "synthesize"],
        "has_synthesis": True,
    },
}


# ════════════════════════════════════════════════════════════════════════
# EPISODE EXECUTION
# ════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeRecord:
    # Design
    gamma: float
    model_config: str
    topology: str
    task_id: str
    task_domain: str
    task_difficulty: str
    # Framework assignment
    frameworks_assigned: list[str]
    diversity_score: float
    framework_coverage: float
    # Agent outputs
    roles: list[str]
    models_used: list[str]
    solutions: list[str]
    # Outcome
    final_answer: str
    ground_truth: str
    correct: bool
    # Cost
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    wall_time_s: float
    per_agent_cost: list[float] = field(default_factory=list)
    # Raw
    agent_outputs_raw: list[str] = field(default_factory=list)


def run_episode(
    task,
    gamma: float,
    model_config_name: str,
    model_config: dict,
    topology_name: str,
    topology: dict,
    clients: dict[str, LLMClient],
    seed: int,
) -> EpisodeRecord:
    """Run a single episode with given configuration."""
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

    # Step 2: Generate outputs per role
    solutions = []
    models_used = []
    per_agent_cost = []
    agent_outputs_raw = []
    total_in = 0
    total_out = 0
    total_cost = 0.0

    propose_outputs = []

    for i, (role, fw) in enumerate(zip(roles, assignments)):
        # Select model for this role
        if role == "synthesize":
            model_key = model_config.get("synthesize", model_config["propose"])
        elif role == "critique":
            model_key = model_config.get("critique", model_config["propose"])
        else:  # propose, insight
            model_key = model_config["propose"]

        client = clients[model_key]

        # Build prompt based on role
        if role == "propose":
            sys_prompt = task.framework_prompts.get(fw, f"Solve using the {fw} approach.")
            usr_prompt = (
                f"Problem: {task.problem}\n\n"
                f"Solve step by step using the {fw} approach.\n"
                f"State your final answer on a line starting with 'ANSWER: '."
            )
        elif role == "insight":
            # Information-shielded: doesn't see other proposals
            sys_prompt = (
                "You are an independent analyst. Other agents have proposed solutions "
                "but you should NOT rely on them. Generate a completely fresh approach."
            )
            usr_prompt = (
                f"Problem: {task.problem}\n\n"
                f"Context: {len(propose_outputs)} other agents proposed solutions. "
                f"You don't know what they said. Use a fresh approach.\n"
                f"State your final answer on a line starting with 'ANSWER: '."
            )
        elif role == "critique":
            # See proposals and find flaws
            proposals_text = "\n".join(
                f"Proposal {j+1}: {p}" for j, p in enumerate(propose_outputs)
            ) if propose_outputs else "No proposals yet."
            sys_prompt = (
                "You are a critical reviewer. Find flaws in the proposals, "
                "then state what you think the correct answer is."
            )
            usr_prompt = (
                f"Problem: {task.problem}\n\n"
                f"Proposals from other agents:\n{proposals_text}\n\n"
                f"Critique these proposals. Then state YOUR answer on a line starting with 'ANSWER: '."
            )
        elif role == "synthesize":
            # See all outputs and synthesize
            all_text = "\n".join(
                f"Agent {j+1} ({roles[j]}, {assignments[j]}): {s}"
                for j, s in enumerate(solutions) if j < i
            )
            sys_prompt = (
                "You are a senior synthesizer. Review all agent outputs, "
                "identify the best answer, and produce the final recommendation."
            )
            usr_prompt = (
                f"Problem: {task.problem}\n\n"
                f"Agent outputs:\n{all_text}\n\n"
                f"Synthesize into a final answer. State it on a line starting with 'ANSWER: '."
            )
        else:
            raise ValueError(f"Unknown role: {role}")

        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        answer = _extract_answer(resp.content)
        solutions.append(answer)
        models_used.append(model_key)
        agent_outputs_raw.append(resp.content[:500])  # truncate for storage

        if role in ("propose", "insight"):
            propose_outputs.append(resp.content[:300])

        total_in += resp.input_tokens
        total_out += resp.output_tokens
        total_cost += resp.cost_usd
        per_agent_cost.append(resp.cost_usd)

    # Step 3: Determine final answer
    if has_synthesis and solutions:
        # Use synthesizer's answer (last agent)
        final_answer = solutions[-1]
    else:
        # Majority vote across all solutions
        counter = Counter(solutions)
        final_answer = counter.most_common(1)[0][0]

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
# MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════════════════

def count_conditions():
    """Count total experimental conditions for cost estimation."""
    n = 0
    for mc_name in MODEL_CONFIGS:
        for tp_name in TOPOLOGIES:
            for gamma in GAMMA_VALUES:
                n += EPISODES_PER_CONDITION
    return n, n * N_AGENTS


def run_experiment(dry_run: bool = False) -> list[dict]:
    n_episodes, n_calls = count_conditions()

    print("=" * 70)
    print("COMPREHENSIVE FACTORIAL EXPERIMENT")
    print("=" * 70)
    print(f"  γ values:       {GAMMA_VALUES}")
    print(f"  Model configs:  {list(MODEL_CONFIGS.keys())}")
    print(f"  Topologies:     {list(TOPOLOGIES.keys())}")
    print(f"  Episodes/cond:  {EPISODES_PER_CONDITION}")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Total API calls: {n_calls}")
    print()

    if dry_run:
        # Estimate cost: ~800 tokens/call, weighted by model pricing
        # Most calls are qwen-7b ($0.30/M), some are qwen-22b ($0.40/M), few qwen-17b ($2.10/M)
        configs_with_17b = sum(1 for mc in MODEL_CONFIGS.values() if "qwen-17b" in mc.values())
        configs_with_22b = sum(1 for mc in MODEL_CONFIGS.values() if "qwen-22b" in mc.values())
        configs_total = len(MODEL_CONFIGS)

        frac_17b = configs_with_17b / configs_total * 0.3  # ~30% of calls in those configs
        frac_22b = configs_with_22b / configs_total * 0.5
        frac_7b = 1.0 - frac_17b - frac_22b

        avg_cost_per_call = (
            frac_7b * 800 * 0.30 / 1_000_000 +
            frac_22b * 800 * 0.40 / 1_000_000 +
            frac_17b * 800 * 2.10 / 1_000_000
        )
        est_cost = n_calls * avg_cost_per_call
        est_minutes = n_calls * 1.5 / 60  # ~1.5s per call average

        print(f"[DRY RUN] Estimated cost:  ${est_cost:.2f}")
        print(f"[DRY RUN] Estimated time:  {est_minutes:.0f} minutes")
        print(f"[DRY RUN] Cost breakdown:")
        print(f"  7B calls (~{frac_7b:.0%}):  ${n_calls * frac_7b * 800 * 0.30 / 1e6:.2f}")
        print(f"  22B calls (~{frac_22b:.0%}): ${n_calls * frac_22b * 800 * 0.40 / 1e6:.2f}")
        print(f"  17B calls (~{frac_17b:.0%}): ${n_calls * frac_17b * 800 * 2.10 / 1e6:.2f}")
        return []

    # Create clients (one per model, reused across conditions)
    needed_models = set()
    for mc in MODEL_CONFIGS.values():
        needed_models.update(v for k, v in mc.items() if k != "description")
    # Always include all model sizes for baselines
    for m in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        needed_models.add(m)
    clients = {m: LLMClient(model_key=m) for m in needed_models}
    print(f"Initialized clients: {list(clients.keys())}")

    rng = random.Random(42)
    all_records: list[EpisodeRecord] = []
    baseline_records: list[dict] = []
    t_start = time.time()
    cumulative_cost = 0.0

    # ── BASELINES: Single model references ─────────────────────────────
    # Run each Qwen model size as a single agent on ALL tasks (no game, no γ)
    # This is the reference: can the game beat a single frontier model?
    print("\n--- BASELINES: Single model per task ---")
    for model_key in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        client = clients[model_key]
        correct_count = 0
        total_baseline_cost = 0.0
        for task in PHASE_A_TASKS:
            try:
                sys_prompt = "You are a problem solver. Solve step by step. State your final answer on a line starting with 'ANSWER: '."
                usr_prompt = f"Problem: {task.problem}"
                resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                answer = _extract_answer(resp.content)
                correct = task.check(answer)
            except Exception as e:
                print(f"    [ERROR] baseline {model_key}/{task.id}: {e}")
                answer, correct, resp = "", False, type("R", (), {"cost_usd": 0, "input_tokens": 0, "output_tokens": 0})()
            if correct:
                correct_count += 1
            total_baseline_cost += resp.cost_usd
            baseline_records.append({
                "model": model_key,
                "task_id": task.id,
                "answer": answer,
                "correct": correct,
                "cost_usd": resp.cost_usd,
                "tokens": resp.input_tokens + resp.output_tokens,
            })
        rate = correct_count / len(PHASE_A_TASKS)
        cumulative_cost += total_baseline_cost
        print(f"  {model_key:12s}: {rate:.1%} correct ({correct_count}/{len(PHASE_A_TASKS)}) — ${total_baseline_cost:.4f}")

    # ── MAIN SWEEP ─────────────────────────────────────────────────────
    condition_idx = 0
    total_conditions = len(MODEL_CONFIGS) * len(TOPOLOGIES) * len(GAMMA_VALUES)

    for mc_name, mc in MODEL_CONFIGS.items():
        for tp_name, tp in TOPOLOGIES.items():
            for gamma in GAMMA_VALUES:
                condition_idx += 1
                # Sample tasks for this condition
                tasks = rng.choices(PHASE_A_TASKS, k=EPISODES_PER_CONDITION)

                cond_records = []
                for ep_idx, task in enumerate(tasks):
                    try:
                        record = run_episode(
                            task=task,
                            gamma=gamma,
                            model_config_name=mc_name,
                            model_config=mc,
                            topology_name=tp_name,
                            topology=tp,
                            clients=clients,
                            seed=rng.randint(0, 2**31),
                        )
                        cond_records.append(record)
                        cumulative_cost += record.total_cost_usd
                    except Exception as e:
                        print(f"    [ERROR] ep {ep_idx} {mc_name}/{tp_name}/γ={gamma}: {e}")
                        continue  # skip this episode, don't kill the run

                all_records.extend(cond_records)

                # Progress report
                if not cond_records:
                    print(f"  [{condition_idx:3d}/{total_conditions}] {mc_name:20s} {tp_name:22s} γ={gamma:4.1f} | ALL EPISODES FAILED")
                    continue
                mean_d = sum(r.diversity_score for r in cond_records) / len(cond_records)
                mean_correct = sum(r.correct for r in cond_records) / len(cond_records)
                mean_cost = sum(r.total_cost_usd for r in cond_records) / len(cond_records)

                elapsed = time.time() - t_start
                print(
                    f"  [{condition_idx:3d}/{total_conditions}] "
                    f"{mc_name:20s} {tp_name:22s} γ={gamma:4.1f} | "
                    f"D={mean_d:.2f} correct={mean_correct:.0%} "
                    f"cost/ep=${mean_cost:.4f} "
                    f"cumul=${cumulative_cost:.3f} [{elapsed:.0f}s]"
                )

    # ── Save complete results ──────────────────────────────────────────
    elapsed = time.time() - t_start
    output_dir = RESULTS_DIR / "full_sweep"
    output_dir.mkdir(exist_ok=True)

    # Full data
    data = {
        "experiment": "full_factorial_sweep",
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
    print(f"\nFull results saved to: {path}")

    # ── Aggregated analysis ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*70}")

    # 1. Diversity vs γ (across all configs)
    # 0. Baselines recap
    print(f"\n--- BASELINE: Single model references ---")
    for model_key in ["qwen-7b", "qwen-22b", "qwen-17b"]:
        br = [b for b in baseline_records if b["model"] == model_key]
        rate = sum(b["correct"] for b in br) / len(br) if br else 0
        cost = sum(b["cost_usd"] for b in br) if br else 0
        print(f"  {model_key:12s}: {rate:.1%} correct, total ${cost:.4f}")

    # 1. Diversity vs γ
    print(f"\n--- Diversity D(γ) across all configurations ---")
    print(f"{'γ':>6} {'mean D':>8} {'correct':>8} {'cost/ep':>10}")
    for gamma in GAMMA_VALUES:
        eps = [r for r in all_records if r.gamma == gamma]
        md = sum(r.diversity_score for r in eps) / len(eps)
        mc = sum(r.correct for r in eps) / len(eps)
        mcost = sum(r.total_cost_usd for r in eps) / len(eps)
        print(f"{gamma:>6.1f} {md:>8.3f} {mc:>8.1%} {mcost:>10.5f}")

    # 2. By model config
    print(f"\n--- Correctness by model config (aggregated over γ, topology) ---")
    for mc_name in MODEL_CONFIGS:
        eps = [r for r in all_records if r.model_config == mc_name]
        mc = sum(r.correct for r in eps) / len(eps)
        mcost = sum(r.total_cost_usd for r in eps) / len(eps)
        print(f"  {mc_name:25s}: correct={mc:.1%}  cost/ep=${mcost:.5f}")

    # 3. By topology
    print(f"\n--- Correctness by topology (aggregated over γ, model) ---")
    for tp_name in TOPOLOGIES:
        eps = [r for r in all_records if r.topology == tp_name]
        mc = sum(r.correct for r in eps) / len(eps)
        mcost = sum(r.total_cost_usd for r in eps) / len(eps)
        print(f"  {tp_name:25s}: correct={mc:.1%}  cost/ep=${mcost:.5f}")

    # 4. Cost dynamics: cumulative cost over time
    print(f"\n--- Cost dynamics ---")
    print(f"  Total cost:   ${cumulative_cost:.4f}")
    print(f"  Total time:   {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Cost/episode: ${cumulative_cost / len(all_records):.5f}")

    # Per-model cost breakdown
    print(f"\n--- Per-model cost breakdown ---")
    for model_key, client in clients.items():
        print(f"  {model_key:12s}: ${client.tracker.total_cost:.4f} ({client.tracker.n_calls} calls, {client.tracker.total_tokens:,} tokens)")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE — analyze with: pivot-a/results/full_sweep/results.json")
    print(f"{'='*70}")

    return [asdict(r) for r in all_records]


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_experiment(dry_run=dry_run)
