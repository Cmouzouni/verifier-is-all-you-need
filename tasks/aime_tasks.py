"""AIME-2024 / AIME-2025 loader.

AIME (American Invitational Mathematics Examination) problems have integer
answers in [0, 999]. Each year has 30 problems split across two papers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIMETask:
    id: str
    problem: str
    answer: int  # ground-truth integer answer in [0, 999]
    solution: str = ""  # optional reference solution


def load_aime_tasks(year: int = 2024, n: int = -1, seed: int = 42) -> list[AIMETask]:
    """Load AIME problems from a year-specific HuggingFace dataset.

    Args:
        year: 2024 or 2025
        n: number of problems to load (-1 = all 30)
        seed: random seed for sampling when n < 30
    """
    from datasets import load_dataset
    import random

    if year == 2024:
        ds_name = "Maxwell-Jia/AIME_2024"
        split = "train"
    elif year == 2025:
        ds_name = "yentinglin/aime_2025"
        split = "train"
    else:
        raise ValueError(f"unsupported AIME year: {year}")

    ds = load_dataset(ds_name, split=split)
    if n < 0 or n >= len(ds):
        indices = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(ds)), n))

    tasks = []
    for i in indices:
        item = ds[i]
        if year == 2024:
            tasks.append(AIMETask(
                id=item.get("ID") or f"aime{year}_{i:03d}",
                problem=item["Problem"],
                answer=int(item["Answer"]),
                solution=item.get("Solution", ""),
            ))
        else:  # 2025 — schema may differ; handle defensively
            problem = item.get("problem") or item.get("Problem") or item.get("question", "")
            answer = item.get("answer") or item.get("Answer")
            if isinstance(answer, str):
                # strip everything but digits
                import re
                m = re.search(r"-?\d+", answer)
                answer = int(m.group(0)) if m else -1
            tasks.append(AIMETask(
                id=item.get("id") or item.get("ID") or f"aime{year}_{i:03d}",
                problem=problem,
                answer=int(answer) if answer is not None else -1,
                solution=item.get("solution") or item.get("Solution", ""),
            ))
    return tasks


if __name__ == "__main__":
    tasks = load_aime_tasks(year=2024, n=3)
    for t in tasks:
        print(f"{t.id}: answer={t.answer}")
        print(f"  {t.problem[:200]}...")
        print()
