"""Extended framework battery: K=8 frameworks per task with informative value priors.

The 4 GOOD frameworks (V=1.0) are the original Phase A frameworks that produce
correct answers when applied properly.

The 4 NOISY frameworks (V=0.3) are intentionally vague or incomplete approaches
that the LLM will attempt but often produce wrong answers because the framework
itself doesn't lead to a verifiable solution.

This is the test bed for MFG vs round-robin: with N=4 < K=8, the choice of
which 4 frameworks to cover matters. MFG should preferentially select the
high-value frameworks; random selection has no such advantage.
"""

from __future__ import annotations

from .phase_a_tasks import (
    PHASE_A_TASKS, Task,
    MATH_FRAMEWORKS, MATH_FRAMEWORK_PROMPTS,
    PATTERN_FRAMEWORKS, PATTERN_FRAMEWORK_PROMPTS,
    LOGIC_FRAMEWORKS, LOGIC_FRAMEWORK_PROMPTS,
    WORD_PROBLEM_FRAMEWORKS, WORD_PROBLEM_FRAMEWORK_PROMPTS,
)


# ── Noisy framework prompts (intentionally low-quality approaches) ──────
NOISY_FRAMEWORK_PROMPTS = {
    "guess_check_random": (
        "You are a problem solver. Solve this by GUESSING a random number first, "
        "then checking if it works. If not, guess another. Don't reason about "
        "the structure of the problem—just guess and check."
    ),
    "intuition_only": (
        "You are a problem solver. Solve this using ONLY YOUR INTUITION. Do not "
        "compute, do not set up equations, do not analyze. Just say what your "
        "gut tells you the answer is. Trust your first instinct."
    ),
    "vague_estimate": (
        "You are a problem solver. Solve this with a VAGUE ESTIMATE. Don't be "
        "precise—give a rough, approximate answer based on a quick mental "
        "impression. Don't double-check, don't verify."
    ),
    "narrative_only": (
        "You are a problem solver. Solve this by describing the SCENARIO IN STORY FORM. "
        "Focus on the narrative and characters, not on the numbers or computation. "
        "At the end, give an answer based on what feels right for the story."
    ),
}

NOISY_FRAMEWORKS = list(NOISY_FRAMEWORK_PROMPTS.keys())  # 4 noisy frameworks

# ── Build extended frameworks per domain ────────────────────────────────
EXTENDED_MATH_FRAMEWORKS = MATH_FRAMEWORKS + NOISY_FRAMEWORKS  # 4 + 4 = 8
EXTENDED_PATTERN_FRAMEWORKS = PATTERN_FRAMEWORKS + NOISY_FRAMEWORKS
EXTENDED_LOGIC_FRAMEWORKS = LOGIC_FRAMEWORKS + NOISY_FRAMEWORKS
EXTENDED_WORD_PROBLEM_FRAMEWORKS = WORD_PROBLEM_FRAMEWORKS + NOISY_FRAMEWORKS

EXTENDED_MATH_PROMPTS = {**MATH_FRAMEWORK_PROMPTS, **NOISY_FRAMEWORK_PROMPTS}
EXTENDED_PATTERN_PROMPTS = {**PATTERN_FRAMEWORK_PROMPTS, **NOISY_FRAMEWORK_PROMPTS}
EXTENDED_LOGIC_PROMPTS = {**LOGIC_FRAMEWORK_PROMPTS, **NOISY_FRAMEWORK_PROMPTS}
EXTENDED_WORD_PROBLEM_PROMPTS = {**WORD_PROBLEM_FRAMEWORK_PROMPTS, **NOISY_FRAMEWORK_PROMPTS}


# Domain → (frameworks_list, prompts_dict, value_priors_dict)
EXTENDED_FRAMEWORK_REGISTRY = {
    "math": {
        "frameworks": EXTENDED_MATH_FRAMEWORKS,
        "prompts": EXTENDED_MATH_PROMPTS,
        "values": {**{f: 1.0 for f in MATH_FRAMEWORKS}, **{f: 0.3 for f in NOISY_FRAMEWORKS}},
    },
    "pattern": {
        "frameworks": EXTENDED_PATTERN_FRAMEWORKS,
        "prompts": EXTENDED_PATTERN_PROMPTS,
        "values": {**{f: 1.0 for f in PATTERN_FRAMEWORKS}, **{f: 0.3 for f in NOISY_FRAMEWORKS}},
    },
    "logic": {
        "frameworks": EXTENDED_LOGIC_FRAMEWORKS,
        "prompts": EXTENDED_LOGIC_PROMPTS,
        "values": {**{f: 1.0 for f in LOGIC_FRAMEWORKS}, **{f: 0.3 for f in NOISY_FRAMEWORKS}},
    },
    "word_problem": {
        "frameworks": EXTENDED_WORD_PROBLEM_FRAMEWORKS,
        "prompts": EXTENDED_WORD_PROBLEM_PROMPTS,
        "values": {**{f: 1.0 for f in WORD_PROBLEM_FRAMEWORKS}, **{f: 0.3 for f in NOISY_FRAMEWORKS}},
    },
}


def make_extended_tasks() -> list[Task]:
    """Convert PHASE_A_TASKS to extended versions with K=8 frameworks per task."""
    extended = []
    for t in PHASE_A_TASKS:
        reg = EXTENDED_FRAMEWORK_REGISTRY[t.domain]
        new_task = Task(
            id=t.id,
            problem=t.problem,
            ground_truth=t.ground_truth,
            domain=t.domain,
            difficulty=t.difficulty,
            frameworks=reg["frameworks"],
            framework_prompts=reg["prompts"],
        )
        extended.append(new_task)
    return extended


def get_value_priors(task: Task) -> dict[str, float]:
    """Get the per-framework value priors for a task (used by MFG)."""
    return EXTENDED_FRAMEWORK_REGISTRY[task.domain]["values"]


EXTENDED_PHASE_A_TASKS = make_extended_tasks()


if __name__ == "__main__":
    tasks = EXTENDED_PHASE_A_TASKS
    print(f"Extended tasks: {len(tasks)}")
    t = tasks[0]
    print(f"\nExample task: {t.id}")
    print(f"  Frameworks (K={len(t.frameworks)}):")
    values = get_value_priors(t)
    for f in t.frameworks:
        v = values[f]
        marker = "★" if v > 0.5 else " "
        print(f"    {marker} {f:25s} V={v}")
