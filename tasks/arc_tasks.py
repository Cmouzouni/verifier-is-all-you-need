"""ARC-AGI loader and grid formatter for the capability floor experiment.

ARC-AGI (Abstraction and Reasoning Corpus, Chollet 2019) tests visual-spatial
rule induction. Each task contains a few input-output grid pairs as training
examples; the model must induce the underlying transformation rule and apply
it to a held-out test input.

We format grids as text (numbers separated by spaces, rows on separate lines)
and use exact grid match for verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ARCTask:
    id: str
    train_pairs: list[tuple[list[list[int]], list[list[int]]]]  # (input, output) pairs
    test_input: list[list[int]]
    test_output: list[list[int]]  # ground truth

    def format_prompt(self) -> str:
        """Format the task as a text prompt."""
        lines = ["Here are some input-output examples of a transformation rule:"]
        for i, (inp, out) in enumerate(self.train_pairs):
            lines.append(f"\nExample {i+1}:")
            lines.append("Input:")
            lines.extend(self._grid_to_text(inp).split("\n"))
            lines.append("Output:")
            lines.extend(self._grid_to_text(out).split("\n"))

        lines.append("\nNow apply the same transformation to this input:")
        lines.append("Input:")
        lines.extend(self._grid_to_text(self.test_input).split("\n"))
        lines.append("\nProvide the output grid in the SAME format as the examples (rows of space-separated numbers).")
        lines.append("State your answer on lines starting with 'OUTPUT:' and ending with 'END_OUTPUT'.")
        return "\n".join(lines)

    @staticmethod
    def _grid_to_text(grid: list[list[int]]) -> str:
        return "\n".join(" ".join(str(c) for c in row) for row in grid)

    def check(self, answer: str) -> bool:
        """Check if the answer matches the test_output grid."""
        parsed = parse_grid_answer(answer)
        if parsed is None:
            return False
        if len(parsed) != len(self.test_output):
            return False
        for r1, r2 in zip(parsed, self.test_output):
            if len(r1) != len(r2):
                return False
            if any(a != b for a, b in zip(r1, r2)):
                return False
        return True


def parse_grid_answer(answer: str) -> list[list[int]] | None:
    """Parse a model output into a 2D integer grid.

    Strategy: find the LAST OUTPUT/END_OUTPUT block (so models can mention
    the word "OUTPUT" in their reasoning before producing the actual answer),
    then parse rows of integers from inside that block. Falls back to parsing
    any consistent block of integer rows.
    """
    if not answer:
        return None

    # Strip <think> tags
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)

    # Prefer the LAST OUTPUT...END_OUTPUT pair so we don't catch reasoning
    # text that mentions "OUTPUT" before the actual answer block.
    block = None
    output_end_pairs = list(re.finditer(
        r"OUTPUT:?\s*\n?(.+?)\s*END_?OUTPUT",
        answer,
        re.DOTALL | re.IGNORECASE,
    ))
    if output_end_pairs:
        block = output_end_pairs[-1].group(1)
    else:
        # No END_OUTPUT marker — fall back to "everything after the last OUTPUT:"
        last_output = list(re.finditer(r"OUTPUT:?\s*\n", answer, re.IGNORECASE))
        if last_output:
            block = answer[last_output[-1].end():]
        else:
            # No OUTPUT marker at all — search the whole text
            block = answer

    # Find lines that look like grid rows (integers separated by spaces)
    rows = []
    for line in block.split("\n"):
        line = line.strip().strip("|").strip()
        # Remove markdown formatting
        line = re.sub(r"[*`\[\],]", " ", line)
        # Find all integers
        ints = re.findall(r"-?\d+", line)
        if not ints:
            continue
        try:
            row = [int(x) for x in ints]
            # Reasonable bounds (ARC grids are small)
            if 1 <= len(row) <= 30:
                rows.append(row)
        except ValueError:
            continue

    if not rows:
        return None

    # Filter to consistent row width — take the largest contiguous block of same-width rows
    if len(set(len(r) for r in rows)) > 1:
        # Find the LARGEST contiguous run of same-width rows (this is the
        # actual grid; spurious integer rows in surrounding text are usually
        # broken up by intervening reasoning text).
        best_run: list[list[int]] = []
        current_run: list[list[int]] = []
        current_width: int | None = None
        for r in rows:
            if current_width is None or len(r) == current_width:
                current_run.append(r)
                current_width = len(r)
            else:
                if len(current_run) > len(best_run):
                    best_run = current_run
                current_run = [r]
                current_width = len(r)
        if len(current_run) > len(best_run):
            best_run = current_run
        rows = best_run

    if not rows:
        return None

    return rows


def load_arc_tasks(n: int = 30, seed: int = 42, split: str = "training") -> list[ARCTask]:
    """Load N ARC-AGI-1 tasks from HuggingFace.

    Args:
        n: number of tasks to sample. Pass n=-1 (or any number ≥ split size) to
            load the full split in deterministic order.
        seed: RNG seed used for the random sample (when n < split size).
        split: 'training' (400 tasks) or 'evaluation' (400 tasks). The
            evaluation split is the canonical ARC-AGI-1 public eval used by
            frontier-model leaderboards.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("lordspline/arc-agi", split=split)
    if n < 0 or n >= len(ds):
        # full split, deterministic order
        indices = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), n)

    tasks = []
    for i in indices:
        item = ds[i]
        train_pairs = [(p["input"], p["output"]) for p in item["train"]]
        # ARC tasks can have multiple test pairs; take the first
        test = item["test"][0]
        tasks.append(ARCTask(
            id=f"arc1_{split}_{i:04d}",
            train_pairs=train_pairs,
            test_input=test["input"],
            test_output=test["output"],
        ))
    return tasks


def load_arc2_tasks(n: int = 100, seed: int = 42, split: str = "test") -> list[ARCTask]:
    """Load N ARC-AGI-2 tasks from `sirorezka/arc-agi-2`.

    ARC-AGI-2 launched in 2025 and is materially harder than ARC-AGI-1.
    Frontier reasoners typically score <5% on it. The dataset has:
        - 'train': 1076 tasks
        - 'test':  167 tasks (the canonical eval set we use here)

    Schema differs slightly from ARC-AGI-1:
        examples: list[{'input', 'output'}]   # the training pairs
        question: list[{'input', 'output'}]   # length 1, the test pair
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("sirorezka/arc-agi-2", split=split)
    if n < 0 or n >= len(ds):
        indices = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), n)

    tasks = []
    for i in indices:
        item = ds[i]
        # examples → train_pairs
        train_pairs = [(ex["input"], ex["output"]) for ex in item["examples"]]
        # question → test (always length 1 in this dataset)
        q = item["question"][0]
        tasks.append(ARCTask(
            id=f"arc2_{split}_{i:04d}",
            train_pairs=train_pairs,
            test_input=q["input"],
            test_output=q["output"],
        ))
    return tasks


if __name__ == "__main__":
    tasks = load_arc_tasks(n=3)
    print(f"Loaded {len(tasks)} ARC tasks\n")
    for t in tasks[:1]:
        print(f"Task {t.id}")
        print(f"  Training pairs: {len(t.train_pairs)}")
        print(f"  Test input shape: {len(t.test_input)}x{len(t.test_input[0])}")
        print(f"  Test output shape: {len(t.test_output)}x{len(t.test_output[0])}")
        print(f"\nPrompt preview (first 600 chars):")
        print(t.format_prompt()[:600])
        print(f"\nGround truth output:")
        print(t._grid_to_text(t.test_output))
