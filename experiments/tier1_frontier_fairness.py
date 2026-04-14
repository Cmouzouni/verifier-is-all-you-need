"""Tier 1.4 — Frontier model fairness re-run.

Question: Did the qwen-17b (Qwen3.5-397B-A17B) baseline of 70% suffer from the
/no_think mode and 1024 max_tokens cap? Re-run with thinking enabled and longer
budget.

The original baseline used:
  - /no_think appended to system prompt (forces non-thinking mode)
  - max_tokens=1024

This re-run uses:
  - NO /no_think (let the model think freely)
  - max_tokens=4096 (room for both thinking and answer)
  - Otherwise identical: same 30 Phase A tasks, same prompts, same temperature

We also re-run on GSM8K-50 to test under the harder benchmark.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from src.config import RESULTS_DIR, MODELS, COST_PER_M_INPUT, COST_PER_M_OUTPUT, TOGETHER_API_KEY
from src.runner import _extract_answer
from tasks.phase_a_tasks import PHASE_A_TASKS
from tasks.gsm8k_tasks import load_gsm8k


# ── Direct OpenAI client (bypass our wrapper to avoid /no_think injection) ─
def make_raw_client():
    return OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")


def call_frontier_thinking(client, system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """Direct call WITHOUT /no_think; let the model think freely."""
    try:
        resp = client.chat.completions.create(
            model=MODELS["qwen-17b"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content or ""
        # Strip think tags but preserve content
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not clean and raw:
            # All in think tags — extract from inside
            m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if m:
                clean = m.group(1).strip()
        usage = resp.usage
        cost = (usage.prompt_tokens * COST_PER_M_INPUT["qwen-17b"] +
                usage.completion_tokens * COST_PER_M_OUTPUT["qwen-17b"]) / 1e6
        return {
            "content": clean if clean else raw,
            "raw_content": raw,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost": cost,
        }
    except Exception as e:
        return {"content": "", "raw_content": "", "input_tokens": 0, "output_tokens": 0, "cost": 0, "error": str(e)[:120]}


def run_baseline_thinking(client, task, max_tokens: int) -> dict:
    sys_prompt = "You are a problem solver. Think step by step. State your final answer on a line starting with 'ANSWER: '."
    usr_prompt = f"Problem: {task.problem}"
    result = call_frontier_thinking(client, sys_prompt, usr_prompt, max_tokens)
    answer = _extract_answer(result["content"])
    return {
        "task_id": task.id,
        "answer": answer,
        "correct": task.check(answer),
        "cost": result["cost"],
        "tokens_in": result["input_tokens"],
        "tokens_out": result["output_tokens"],
        "raw_preview": result["content"][:200],
    }


def main(dry_run: bool = False):
    print("=" * 70)
    print("TIER 1.4 — FRONTIER FAIRNESS RE-RUN")
    print("=" * 70)

    print(f"\nModel: qwen-17b ({MODELS['qwen-17b']}) WITH thinking enabled")
    print(f"Max tokens: 4096 (vs original 1024)")

    if dry_run:
        n_calls = 30 + 50  # Phase A + GSM8K
        # Assume avg 3000 tokens output (thinking heavy)
        est = n_calls * 3500 * 3.60 / 1e6  # output cost dominates
        print(f"\n[DRY RUN] {n_calls} calls, est ${est:.2f}")
        return

    client = make_raw_client()
    t_start = time.time()
    output_dir = RESULTS_DIR / "tier1_frontier"
    output_dir.mkdir(exist_ok=True)

    # ── Phase A tasks ──────────────────────────────────────────────────
    print(f"\n--- Phase A tasks ({len(PHASE_A_TASKS)}) ---", flush=True)
    phase_a_results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_baseline_thinking, client, t, 4096) for t in PHASE_A_TASKS]
        completed = 0
        for f in as_completed(futures):
            phase_a_results.append(f.result())
            completed += 1
            if completed % 5 == 0:
                k = sum(r["correct"] for r in phase_a_results)
                cost = sum(r["cost"] for r in phase_a_results)
                print(f"  [{completed}/{len(PHASE_A_TASKS)}] {k}/{completed} correct, ${cost:.4f}", flush=True)

    phase_a_rate = sum(r["correct"] for r in phase_a_results) / len(phase_a_results)
    phase_a_cost = sum(r["cost"] for r in phase_a_results)
    print(f"\nPhase A re-run: {phase_a_rate:.1%} correct (vs original 70.0%)")
    print(f"Phase A cost: ${phase_a_cost:.4f}")

    # ── GSM8K subset ───────────────────────────────────────────────────
    print(f"\n--- GSM8K-50 ---", flush=True)
    gsm = load_gsm8k(n=50, seed=42)
    gsm_results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_baseline_thinking, client, t, 4096) for t in gsm]
        completed = 0
        for f in as_completed(futures):
            gsm_results.append(f.result())
            completed += 1
            if completed % 10 == 0:
                k = sum(r["correct"] for r in gsm_results)
                cost = sum(r["cost"] for r in gsm_results)
                print(f"  [{completed}/{len(gsm)}] {k}/{completed} correct, ${cost:.4f}", flush=True)

    gsm_rate = sum(r["correct"] for r in gsm_results) / len(gsm_results)
    gsm_cost = sum(r["cost"] for r in gsm_results)
    print(f"\nGSM8K-50: {gsm_rate:.1%} correct")
    print(f"GSM8K cost: ${gsm_cost:.4f}")

    elapsed = time.time() - t_start
    total_cost = phase_a_cost + gsm_cost

    print(f"\n{'='*70}")
    print(f"TIER 1.4 RESULTS — Frontier Fairness")
    print(f"{'='*70}")
    print(f"\n  qwen-17b ORIGINAL (no_think, 1024 tok):  70.0% on Phase A")
    print(f"  qwen-17b RE-RUN  (think, 4096 tok):     {phase_a_rate:.1%} on Phase A  cost=${phase_a_cost:.4f}")
    print(f"  qwen-17b RE-RUN  (think, 4096 tok):     {gsm_rate:.1%} on GSM8K-50  cost=${gsm_cost:.4f}")
    print(f"\n  Average cost per problem: ${total_cost / (len(phase_a_results) + len(gsm_results)):.5f}")
    print(f"  Δ vs original Phase A 70%: {phase_a_rate - 0.70:+.1%}")
    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s")

    # Save
    json.dump({
        "experiment": "tier1_frontier_fairness",
        "model": "qwen-17b",
        "settings": {"thinking": True, "max_tokens": 4096},
        "phase_a": {
            "n": len(phase_a_results),
            "rate": phase_a_rate,
            "cost_total": phase_a_cost,
            "cost_per": phase_a_cost / len(phase_a_results),
            "results": phase_a_results,
        },
        "gsm8k_50": {
            "n": len(gsm_results),
            "rate": gsm_rate,
            "cost_total": gsm_cost,
            "cost_per": gsm_cost / len(gsm_results),
            "results": gsm_results,
        },
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
