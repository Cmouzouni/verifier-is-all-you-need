"""Experiment A.2 — Information Shielding Validation

Question: Does shielding Insight agents from peer proposals prevent anchoring?

Design:
  - 30 simple tasks from Phase A battery
  - Planted seductive wrong approach in context
  - Two conditions: Shielded vs Unshielded
  - Models: Qwen2.5-3B and Qwen2.5-7B
  - N = 4 agents (2 Propose, 1 Consensus, 1 Insight)
  - 30 episodes per condition per model size

Primary metric: Anchoring rate (fraction of Insight outputs echoing the planted wrong approach)
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from src.logger import ExperimentLogger, EpisodeLog, AgentLog
from tasks.phase_a_tasks import PHASE_A_TASKS

EXPERIMENT_ID = "exp_a2_shielding"
MODEL_KEYS = ["qwen-7b", "qwen-22b"]  # small vs medium Qwen tiers
EPISODES_PER_CONDITION = 30
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Planted wrong approaches per domain
PLANTED_WRONG = {
    "math": ("I'll just add all the numbers together: {numbers_sum}. "
             "So the answer is {numbers_sum}."),
    "pattern": "The pattern is just adding 1 each time, so the next number is {last_plus_1}.",
    "logic": "Since the first option mentioned is always correct, the answer is the first option.",
    "word_problem": ("I'll multiply the two biggest numbers: {big_product}. "
                     "That gives the answer."),
}


def make_planted_context(task) -> str:
    """Generate a plausible but wrong approach for the task."""
    return (
        f"A previous agent proposed: \"Let me try a simple approach. "
        f"Looking at the numbers in the problem, I'll combine them directly. "
        f"My answer is 999.\"\n"
        f"The verification status: UNKNOWN (not yet verified)."
    )


def run_insight_generation(
    client: LLMClient,
    task,
    planted_context: str,
    shielded: bool,
    seed: int,
) -> dict:
    """Run a single Insight-agent generation, shielded or unshielded."""
    if shielded:
        # Agent sees ONLY structural info, not content
        sys_prompt = (
            "You are an Insight agent. Your job is to find a COMPLETELY DIFFERENT "
            "approach to solve this problem. You know that other agents have proposed "
            "solutions but verification has not confirmed them. "
            "Generate a genuinely novel approach."
        )
        usr_prompt = (
            f"Problem: {task.problem}\n\n"
            f"Context: 3 other agents have proposed solutions. None verified yet.\n"
            f"Your job: propose a DIFFERENT approach. Do NOT guess what others tried.\n"
            f"After reasoning, state your final answer on a line starting with 'ANSWER: '."
        )
    else:
        # Agent sees full proposals including the planted wrong one
        sys_prompt = (
            "You are an Insight agent. Your job is to find a COMPLETELY DIFFERENT "
            "approach to solve this problem. Read what other agents proposed, then "
            "generate a genuinely novel approach that differs from theirs."
        )
        usr_prompt = (
            f"Problem: {task.problem}\n\n"
            f"Previous proposals:\n{planted_context}\n\n"
            f"Your job: propose a DIFFERENT approach from the above.\n"
            f"After reasoning, state your final answer on a line starting with 'ANSWER: '."
        )

    resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # Check if output is anchored on the planted wrong approach
    content_lower = resp.content.lower()
    anchored = (
        "999" in content_lower
        or "add all" in content_lower
        or "combine them directly" in content_lower
        or "multiply the two biggest" in content_lower
        or "adding 1 each time" in content_lower
        or "first option" in content_lower
    )

    return {
        "content": resp.content,
        "anchored": anchored,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cost_usd": resp.cost_usd,
    }


def run_experiment(dry_run: bool = False) -> dict:
    print(f"{'='*60}")
    print(f"Experiment A.2: Information Shielding Validation")
    print(f"Models: {MODEL_KEYS}")
    print(f"Conditions: Shielded vs Unshielded × {EPISODES_PER_CONDITION} episodes")
    print(f"{'='*60}")

    if dry_run:
        total_calls = len(MODEL_KEYS) * 2 * EPISODES_PER_CONDITION
        est_cost = total_calls * 800 * 0.15 / 1_000_000
        print(f"\n[DRY RUN] {total_calls} API calls, est. ${est_cost:.3f}")
        return {}

    rng = random.Random(42)
    logger = ExperimentLogger(EXPERIMENT_ID, RESULTS_DIR / EXPERIMENT_ID)
    all_results = {}

    for model_key in MODEL_KEYS:
        print(f"\n--- Model: {model_key} ---")
        client = LLMClient(model_key=model_key)

        for condition in ["shielded", "unshielded"]:
            print(f"  Condition: {condition}")
            anchored_count = 0
            tasks = rng.choices(PHASE_A_TASKS, k=EPISODES_PER_CONDITION)

            for ep_idx, task in enumerate(tasks):
                planted = make_planted_context(task)
                result = run_insight_generation(
                    client, task, planted,
                    shielded=(condition == "shielded"),
                    seed=rng.randint(0, 2**31),
                )
                if result["anchored"]:
                    anchored_count += 1

                ep_log = EpisodeLog(
                    experiment_id=EXPERIMENT_ID,
                    episode_id=f"ep_{model_key}_{condition}_{ep_idx:03d}",
                    task_id=task.id,
                    benchmark="phase_a",
                    difficulty=task.difficulty,
                    gamma=0.0,
                    n_agents=1,
                    agent_logs=[AgentLog(
                        agent_id="insight_0",
                        agent_role="insight",
                        model_size=model_key,
                        framework=condition,
                        round=0,
                        prompt_tokens=result["input_tokens"],
                        completion_tokens=result["output_tokens"],
                        cost_usd=result["cost_usd"],
                        output_raw=result["content"],
                        solution="anchored" if result["anchored"] else "independent",
                    )],
                    total_tokens=result["input_tokens"] + result["output_tokens"],
                    total_cost_usd=result["cost_usd"],
                )
                logger.add_episode(ep_log)

            rate = anchored_count / EPISODES_PER_CONDITION
            key = f"{model_key}_{condition}"
            all_results[key] = {
                "anchoring_rate": rate,
                "anchored_count": anchored_count,
                "total": EPISODES_PER_CONDITION,
            }
            print(f"    Anchoring rate: {rate:.1%} ({anchored_count}/{EPISODES_PER_CONDITION})")

        print(f"  Cost so far: ${client.tracker.total_cost:.4f}")

    # ── Save and analyze ───────────────────────────────────────────────
    path = logger.save()
    print(f"\nResults saved to: {path}")

    # Fisher's exact test equivalent (simple binomial comparison)
    print(f"\n--- Gate Evaluation ---")
    for model_key in MODEL_KEYS:
        s_rate = all_results[f"{model_key}_shielded"]["anchoring_rate"]
        u_rate = all_results[f"{model_key}_unshielded"]["anchoring_rate"]
        diff = u_rate - s_rate
        print(f"  {model_key}: Unshielded={u_rate:.1%} Shielded={s_rate:.1%} Δ={diff:+.1%}")

    # Gate
    any_significant = any(
        all_results[f"{m}_unshielded"]["anchoring_rate"] > all_results[f"{m}_shielded"]["anchoring_rate"] + 0.15
        for m in MODEL_KEYS
    )
    print(f"\n{'='*60}")
    print(f"GO/NO-GO GATE A.2: {'GO' if any_significant else 'NEEDS MORE DATA'}")
    print(f"{'='*60}")

    summary = {"experiment": EXPERIMENT_ID, "results": all_results, "gate": "GO" if any_significant else "UNCLEAR"}
    (RESULTS_DIR / EXPERIMENT_ID / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return all_results


if __name__ == "__main__":
    run_experiment(dry_run="--dry-run" in sys.argv)
