"""GSM8K benchmark loader for Tier 1 hard-benchmark replication.

Loads the GSM8K test set from HuggingFace, extracts ground-truth answers,
and packages them in the same Task structure used in Phase A.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the same Task structure for compatibility
from .phase_a_tasks import (
    MATH_FRAMEWORKS,
    MATH_FRAMEWORK_PROMPTS,
    Task,
)


def _parse_gsm8k_answer(answer_text: str) -> str:
    """GSM8K answers end with '#### N'. Extract N as a string."""
    m = re.search(r"####\s*(-?\$?[\d,]+\.?\d*)", answer_text)
    if not m:
        return ""
    val = m.group(1).strip().replace(",", "").replace("$", "")
    return val


def load_gsm8k(n: int = 50, seed: int = 42, difficulty: str = "all") -> list[Task]:
    """Load N GSM8K problems as Task objects.

    Args:
        n: Number of problems to sample.
        seed: Random seed for reproducible sampling.
        difficulty: 'all', 'short' (< 50 words), or 'long'.

    Returns:
        List of Task objects with the math framework set.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(seed)

    # Filter by length
    if difficulty == "short":
        items = [x for x in ds if len(x["question"].split()) < 40]
    elif difficulty == "long":
        items = [x for x in ds if len(x["question"].split()) >= 60]
    else:
        items = list(ds)

    # Sample
    sampled = rng.sample(items, min(n, len(items)))

    tasks = []
    for i, item in enumerate(sampled):
        gt = _parse_gsm8k_answer(item["answer"])
        if not gt:
            continue
        tasks.append(Task(
            id=f"gsm8k_{i:03d}",
            problem=item["question"],
            ground_truth=gt,
            domain="math",
            difficulty="medium",
            frameworks=MATH_FRAMEWORKS,
            framework_prompts=MATH_FRAMEWORK_PROMPTS,
        ))

    return tasks


if __name__ == "__main__":
    tasks = load_gsm8k(n=10)
    print(f"Loaded {len(tasks)} GSM8K tasks")
    for t in tasks[:3]:
        print(f"  {t.id}: {t.problem[:80]}... -> {t.ground_truth}")
