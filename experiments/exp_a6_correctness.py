"""Experiment A.6 — Game Correctness Validation

Question: Does the game produce better outputs than a single agent at matched compute cost?

Design:
  - 30 tasks from Phase A battery
  - Three conditions (matched compute budget):
    G: Full game, N=4, γ=1, 3B models
    S: Single 7B model (approximately same cost as 4×3B)
    M: Majority vote, 4 independent 3B agents, no graph structure
  - External verifier throughout
  - 10 episodes per task per condition = 30 × 10 = 300 episodes per condition
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.runner import run_propose_episode, _extract_answer
from src.config import RESULTS_DIR
from src.logger import ExperimentLogger, EpisodeLog, AgentLog
from tasks.phase_a_tasks import PHASE_A_TASKS

EXPERIMENT_ID = "exp_a6_correctness"
EPISODES_PER_TASK = 10
TEMPERATURE = 0.7
MAX_TOKENS = 512


def run_single_agent(client: LLMClient, task, seed: int) -> dict:
    """Single agent baseline: one model, one shot."""
    sys_prompt = "You are a problem solver. Solve the problem step by step. State your final answer on a line starting with 'ANSWER: '."
    usr_prompt = f"Problem: {task.problem}"
    resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    answer = _extract_answer(resp.content)
    correct = task.check(answer)
    return {
        "answer": answer,
        "correct": correct,
        "tokens": resp.input_tokens + resp.output_tokens,
        "cost": resp.cost_usd,
    }


def run_majority_vote(client: LLMClient, task, n_agents: int, seed: int) -> dict:
    """Majority vote: N independent agents, same prompt, take majority answer."""
    rng = random.Random(seed)
    answers = []
    total_tokens = 0
    total_cost = 0.0

    for i in range(n_agents):
        sys_prompt = "You are a problem solver. Solve the problem step by step. State your final answer on a line starting with 'ANSWER: '."
        usr_prompt = f"Problem: {task.problem}"
        resp = client.generate(sys_prompt, usr_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        answer = _extract_answer(resp.content)
        answers.append(answer)
        total_tokens += resp.input_tokens + resp.output_tokens
        total_cost += resp.cost_usd

    # Majority vote (most common answer)
    counter = Counter(answers)
    majority = counter.most_common(1)[0][0]
    correct = task.check(majority)

    return {
        "answer": majority,
        "all_answers": answers,
        "correct": correct,
        "tokens": total_tokens,
        "cost": total_cost,
    }


def run_experiment(dry_run: bool = False) -> dict:
    print(f"{'='*60}")
    print(f"Experiment A.6: Game Correctness Validation")
    print(f"Tasks: {len(PHASE_A_TASKS)} | Episodes/task: {EPISODES_PER_TASK}")
    print(f"Conditions: Game(3B×4,γ=1) vs Single(7B) vs MajVote(3B×4)")
    print(f"{'='*60}")

    n_tasks = len(PHASE_A_TASKS)
    if dry_run:
        # Game: 4 calls per episode. Single: 1 call. MajVote: 4 calls.
        game_calls = n_tasks * EPISODES_PER_TASK * 4
        single_calls = n_tasks * EPISODES_PER_TASK * 1
        mv_calls = n_tasks * EPISODES_PER_TASK * 4
        total = game_calls + single_calls + mv_calls
        est_cost = (game_calls * 800 * 0.08 + single_calls * 800 * 0.14 + mv_calls * 800 * 0.08) / 1_000_000
        print(f"\n[DRY RUN] {total} API calls, est. ${est_cost:.3f}")
        return {}

    client_small = LLMClient(model_key="qwen-7b")    # small Qwen tier
    client_large = LLMClient(model_key="qwen-22b")   # medium Qwen tier
    logger = ExperimentLogger(EXPERIMENT_ID, RESULTS_DIR / EXPERIMENT_ID)
    rng = random.Random(42)

    results = {"game": [], "single": [], "majority": []}
    t_start = time.time()

    for task_idx, task in enumerate(PHASE_A_TASKS):
        if (task_idx + 1) % 10 == 0:
            print(f"  Task {task_idx+1}/{n_tasks}...")

        for ep in range(EPISODES_PER_TASK):
            seed = rng.randint(0, 2**31)

            # Condition G: Game with γ=1
            game_result = run_propose_episode(
                client=client_small, task=task, n_agents=4, gamma=1.0,
                tau=0.3, seed=seed, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
            # Game "correct" = majority of agents correct (or best answer)
            game_answers = game_result.solutions
            game_counter = Counter(game_answers)
            game_majority = game_counter.most_common(1)[0][0]
            game_correct = task.check(game_majority)
            results["game"].append(game_correct)

            # Condition S: Single larger Qwen agent (matched cost ≈ 4×small)
            single_result = run_single_agent(client_large, task, seed)
            results["single"].append(single_result["correct"])

            # Condition M: Majority vote 4×small, no graph structure
            mv_result = run_majority_vote(client_small, task, 4, seed)
            results["majority"].append(mv_result["correct"])

    # ── Analysis ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_cost = client_small.tracker.total_cost + client_large.tracker.total_cost

    game_rate = sum(results["game"]) / len(results["game"])
    single_rate = sum(results["single"]) / len(results["single"])
    mv_rate = sum(results["majority"]) / len(results["majority"])

    print(f"\n--- Results ---")
    print(f"  Game (3B×4, γ=1):    {game_rate:.1%} ({sum(results['game'])}/{len(results['game'])})")
    print(f"  Single (7B):         {single_rate:.1%} ({sum(results['single'])}/{len(results['single'])})")
    print(f"  Majority vote (3B×4): {mv_rate:.1%} ({sum(results['majority'])}/{len(results['majority'])})")
    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Gate evaluation
    print(f"\n{'='*60}")
    print(f"GO/NO-GO GATE A.6:")
    if game_rate > mv_rate + 0.05 and game_rate > single_rate:
        print(f"  ✓ GO: Game ({game_rate:.1%}) > MajVote ({mv_rate:.1%}) and > Single ({single_rate:.1%})")
    elif game_rate >= mv_rate and game_rate >= single_rate:
        print(f"  ~ CONDITIONAL GO: Game matches but doesn't clearly beat baselines")
    elif game_rate >= single_rate:
        print(f"  ~ PARTIAL GO: Game >= Single but MajVote is competitive")
    else:
        print(f"  ✗ NO-GO: Game ({game_rate:.1%}) worse than Single ({single_rate:.1%})")
    print(f"{'='*60}")

    # Save
    summary = {
        "experiment": EXPERIMENT_ID,
        "game_correct_rate": game_rate,
        "single_correct_rate": single_rate,
        "majority_correct_rate": mv_rate,
        "n_episodes": len(results["game"]),
        "total_cost_usd": total_cost,
        "time_s": elapsed,
    }
    (RESULTS_DIR / EXPERIMENT_ID / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / EXPERIMENT_ID / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.save()

    return summary


if __name__ == "__main__":
    run_experiment(dry_run="--dry-run" in sys.argv)
