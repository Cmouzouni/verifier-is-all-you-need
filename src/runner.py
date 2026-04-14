"""Episode runner for Phase A experiments.

Handles: framework assignment with γ → parallel LLM generation → metric collection.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any

from .client import LLMClient
from .congestion import assign_frameworks, FrameworkState
from .schemas import ProposeOutput, ROLE_SCHEMAS
from .diversity import diversity_score, framework_coverage, shannon_entropy
from .logger import EpisodeLog, AgentLog


@dataclass
class EpisodeResult:
    """Result from running one episode."""
    task_id: str
    gamma: float
    assignments: list[str]
    solutions: list[str]
    correct: list[bool]
    diversity: float
    coverage: float
    entropy: float
    population_dist: dict[str, float]
    agent_logs: list[AgentLog]
    total_tokens: int
    total_cost: float
    wall_time: float


def run_propose_episode(
    client: LLMClient,
    task: Any,
    n_agents: int = 4,
    gamma: float = 0.0,
    tau: float = 0.3,
    seed: int | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> EpisodeResult:
    """Run a single Propose-only episode (for A.1 γ response validation).

    1. Sequentially assign frameworks to N agents using congestion mechanism
    2. Each agent generates a solution using their assigned framework
    3. Collect diversity metrics and per-agent logs
    """
    t0 = time.time()

    # Step 1: Framework assignment with congestion
    assignments, state = assign_frameworks(
        frameworks=task.frameworks,
        n_agents=n_agents,
        gamma=gamma,
        tau=tau,
        seed=seed,
    )

    # Step 2: Generate solutions (parallel via batch)
    requests = []
    for i, fw in enumerate(assignments):
        sys_prompt = task.framework_prompts.get(fw, f"Solve this using the {fw} approach.")
        usr_prompt = (
            f"Problem: {task.problem}\n\n"
            f"Solve this step by step using the {fw} approach.\n"
            f"After your reasoning, state your final answer clearly on a line starting with 'ANSWER: '."
        )
        requests.append((sys_prompt, usr_prompt))

    responses = client.generate_batch(requests, temperature=temperature, max_tokens=max_tokens)

    # Step 3: Extract solutions and build logs
    solutions = []
    correct = []
    agent_logs = []
    total_tokens = 0
    total_cost = 0.0

    for i, (resp, fw) in enumerate(zip(responses, assignments)):
        # Extract answer from response
        answer = _extract_answer(resp.content)
        solutions.append(answer)
        is_correct = task.check(answer)
        correct.append(is_correct)

        total_tokens += resp.input_tokens + resp.output_tokens
        total_cost += resp.cost_usd

        agent_logs.append(AgentLog(
            agent_id=f"agent_{i}",
            agent_role="propose",
            model_size=client.model_key,
            framework=fw,
            round=0,
            prompt_tokens=resp.input_tokens,
            completion_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            output_raw=resp.content,
            solution=answer,
            confidence=0.0,  # Not parsed in simple mode
        ))

    wall_time = time.time() - t0

    return EpisodeResult(
        task_id=task.id,
        gamma=gamma,
        assignments=assignments,
        solutions=solutions,
        correct=correct,
        diversity=diversity_score(assignments),
        coverage=framework_coverage(assignments, len(task.frameworks)),
        entropy=shannon_entropy(assignments),
        population_dist=state.occupancy,
        agent_logs=agent_logs,
        total_tokens=total_tokens,
        total_cost=total_cost,
        wall_time=wall_time,
    )


def run_structured_episode(
    client: LLMClient,
    task: Any,
    n_agents: int = 4,
    gamma: float = 0.0,
    tau: float = 0.3,
    seed: int | None = None,
    temperature: float = 0.7,
    max_tokens: int = 768,
) -> EpisodeResult:
    """Run a structured episode with Pydantic schema validation.

    Same as run_propose_episode but uses generate_structured for typed output.
    Used in experiments where we need parsed confidence, approach_description, etc.
    """
    t0 = time.time()

    assignments, state = assign_frameworks(
        frameworks=task.frameworks,
        n_agents=n_agents,
        gamma=gamma,
        tau=tau,
        seed=seed,
    )

    solutions = []
    correct = []
    agent_logs = []
    total_tokens = 0
    total_cost = 0.0

    for i, fw in enumerate(assignments):
        sys_prompt = task.framework_prompts.get(fw, f"Solve this using the {fw} approach.")
        usr_prompt = (
            f"Problem: {task.problem}\n\n"
            f"Solve this using the {fw} approach."
        )
        try:
            parsed, resp = client.generate_structured(
                sys_prompt, usr_prompt, ProposeOutput,
                temperature=temperature, max_tokens=max_tokens,
            )
            answer = parsed.solution
            confidence = parsed.confidence
            schema_valid = True
        except Exception as e:
            # Fallback to unstructured
            resp = client.generate(sys_prompt, usr_prompt, temperature, max_tokens)
            answer = _extract_answer(resp.content)
            confidence = 0.0
            schema_valid = False

        solutions.append(answer)
        is_correct = task.check(answer)
        correct.append(is_correct)
        total_tokens += resp.input_tokens + resp.output_tokens
        total_cost += resp.cost_usd

        agent_logs.append(AgentLog(
            agent_id=f"agent_{i}",
            agent_role="propose",
            model_size=client.model_key,
            framework=fw,
            round=0,
            prompt_tokens=resp.input_tokens,
            completion_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            output_raw=resp.content,
            schema_valid=schema_valid,
            solution=answer,
            confidence=confidence,
        ))

    wall_time = time.time() - t0

    return EpisodeResult(
        task_id=task.id,
        gamma=gamma,
        assignments=assignments,
        solutions=solutions,
        correct=correct,
        diversity=diversity_score(assignments),
        coverage=framework_coverage(assignments, len(task.frameworks)),
        entropy=shannon_entropy(assignments),
        population_dist=state.occupancy,
        agent_logs=agent_logs,
        total_tokens=total_tokens,
        total_cost=total_cost,
        wall_time=wall_time,
    )


def _extract_answer(text: str) -> str:
    """Extract the final answer from LLM output. Robust to many output formats."""
    import re
    # Strip <think>...</think> reasoning tags (Qwen3 models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return ""

    # 1. Look for explicit "ANSWER: ..." pattern (case-insensitive)
    for line in text.split("\n"):
        line = line.strip()
        for prefix in ["ANSWER:", "Answer:", "answer:", "FINAL ANSWER:", "Final Answer:",
                        "**ANSWER:**", "**Answer:**", "**FINAL ANSWER:**"]:
            if line.startswith(prefix):
                val = line[len(prefix):].strip().rstrip(".")
                val = re.sub(r"[\\*$]", "", val).strip()  # strip markdown/latex
                if val:
                    return val

    # 2. Look for "the answer is X" anywhere
    m = re.search(r"[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)", text)
    if m:
        val = re.sub(r"[\\*$]", "", m.group(1)).strip()
        if val:
            return val

    # 3. Look for \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()

    # 4. Look for "= X" at end of a line (math result)
    m = re.search(r"=\s*(.+?)(?:\s*$|\n)", text)
    if m:
        val = m.group(1).strip().rstrip(".")
        val = re.sub(r"[\\*$]", "", val).strip()
        # Only use if it looks like a number or short answer
        if val and len(val) < 30:
            return val

    # 5. Look for bold answer patterns: **5** or **x = 5**
    m = re.search(r"\*\*(?:x\s*=\s*)?(\d+(?:\.\d+)?)\*\*", text)
    if m:
        return m.group(1)

    # 6. Last resort: find the last number in the text
    numbers = re.findall(r"(?<![a-zA-Z])(-?\d+(?:\.\d+)?)(?![a-zA-Z\d])", text)
    if numbers:
        return numbers[-1]

    # 7. Very last resort: last short non-empty line
    for line in reversed(text.strip().split("\n")):
        line = line.strip().rstrip(".")
        if line and len(line) < 50 and not line.startswith("#"):
            for p in ["So ", "Therefore ", "Thus ", "Hence "]:
                if line.startswith(p):
                    line = line[len(p):]
            return line.strip()

    return ""
