"""ARC Cracking Suite — multiple architectures targeting ≥30% on ARC-AGI.

Goal: design and test game architectures that match or exceed the ~30% achieved
by top frontier reasoning models, using a mix of cheap and larger models.

Architectures tested:
  EXP-A1: Massive cheap parallelism (16×qwen-7b) + frontier synth (qwen-17b)
  EXP-A2: Massive cheap parallelism (16×qwen-7b) + medium synth (qwen-22b)
  EXP-B1: Heterogeneous massive (8×qwen-7b + 4×qwen-22b + 2×qwen-17b) + frontier synth
  EXP-C1: Self-consistency with leave-one-out verification (16×qwen-7b, no frontier)
  EXP-C2: Self-consistency with leave-one-out verification (16×qwen-22b, no frontier)
  EXP-D1: Best-of-32 cheap with frontier synth (32×qwen-7b + qwen-17b synth)
  EXP-E1: Tournament — pairwise comparison by frontier model

Plus baselines:
  - Single qwen-7b   (1 attempt, 1 sample)
  - Single qwen-22b
  - Single qwen-17b WITH thinking
  - Best-of-16 qwen-7b (oracle: count if any of 16 samples is correct)
  - Best-of-16 qwen-22b (oracle)
  - Best-of-16 qwen-17b (oracle)

The oracle baselines tell us the THEORETICAL CEILING for any best-of-N approach.

Aggressive parallelism: ThreadPoolExecutor with 30+ workers throughout.
Together AI rate limit is 3000 RPM; we operate well below that.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client import LLMClient
from src.config import RESULTS_DIR
from tasks.arc_tasks import load_arc_tasks, parse_grid_answer, ARCTask


# ── Configuration ──────────────────────────────────────────────────────
N_PROBLEMS = 30
TEMPERATURE = 0.9  # higher temperature for sample diversity
MAX_TOKENS = 2048
WORKERS = 30  # aggressive parallelism — well below 3000 RPM limit


# ════════════════════════════════════════════════════════════════════════
# CORE PRIMITIVES
# ════════════════════════════════════════════════════════════════════════

ARC_SYSTEM_PROMPT = (
    "You are a visual-spatial reasoning expert solving ARC-AGI puzzles. "
    "You will be shown several input-output grid pairs that follow the same "
    "transformation rule. Figure out the rule and apply it to the test input. "
    "Output the resulting grid in the exact same format (rows of space-separated integers) "
    "between OUTPUT: and END_OUTPUT markers."
)


def call_arc(client: LLMClient, task: ARCTask, sys_prompt: str = None, framing: str = None) -> dict:
    """One ARC attempt; returns dict with content and parsed correctness.

    For qwen-17b (frontier thinking model), thinking is enabled automatically.
    """
    sys_p = sys_prompt or ARC_SYSTEM_PROMPT
    if framing:
        sys_p = sys_p + f"\n\nAnalytical strategy: {framing}"
    usr_p = task.format_prompt()
    # Enable thinking for the frontier model — it's the only fair way to use it
    use_thinking = client.model_key == "qwen-17b"
    try:
        resp = client.generate(
            sys_p, usr_p,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            thinking=use_thinking,
        )
        return {
            "content": resp.content,
            "correct": task.check(resp.content),
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        }
    except Exception as e:
        return {"content": "", "correct": False, "tokens": 0, "cost": 0, "error": str(e)[:80]}


def grid_to_canonical(grid: list[list[int]] | None) -> str | None:
    """Convert a 2D grid into a canonical string for equality comparison."""
    if grid is None:
        return None
    return ";".join(",".join(str(c) for c in row) for row in grid)


def parse_canonical(content: str) -> str | None:
    """Parse model output to canonical grid string, or None on failure."""
    g = parse_grid_answer(content)
    return grid_to_canonical(g)


# ════════════════════════════════════════════════════════════════════════
# AGGREGATORS
# ════════════════════════════════════════════════════════════════════════

def aggregate_majority(samples: list[dict]) -> tuple[str | None, int]:
    """Take the most common parsed grid output. Returns (canonical_grid, count)."""
    counts = Counter()
    for s in samples:
        c = parse_canonical(s.get("content", ""))
        if c:
            counts[c] += 1
    if not counts:
        return None, 0
    top, n = counts.most_common(1)[0]
    return top, n


def canonical_to_grid(canonical: str | None) -> list[list[int]] | None:
    if not canonical:
        return None
    rows = []
    for row_str in canonical.split(";"):
        rows.append([int(x) for x in row_str.split(",")])
    return rows


def check_canonical(canonical: str | None, task: ARCTask) -> bool:
    if not canonical:
        return False
    g = canonical_to_grid(canonical)
    if g is None:
        return False
    if len(g) != len(task.test_output):
        return False
    for r1, r2 in zip(g, task.test_output):
        if len(r1) != len(r2) or any(a != b for a, b in zip(r1, r2)):
            return False
    return True


# ════════════════════════════════════════════════════════════════════════
# ARCHITECTURE: SYNTHESIZER OVER N CANDIDATES
# ════════════════════════════════════════════════════════════════════════

def synthesize_over_candidates(client: LLMClient, task: ARCTask, candidates: list[dict]) -> dict:
    """Frontier-class synthesizer reads all candidates and produces final answer.

    Thinking is enabled for the frontier model.
    """
    cand_text = []
    for i, c in enumerate(candidates):
        cand_text.append(f"--- CANDIDATE {i+1} ---\n{c.get('content', '')[:1500]}")
    cand_block = "\n\n".join(cand_text)

    sys_p = (
        "You are a senior ARC-AGI solver. Several proposers analyzed this puzzle "
        "and produced candidate output grids. Read each candidate carefully, identify "
        "which one (if any) correctly applies a transformation rule that's consistent "
        "with the training examples, and produce the final correct output grid. You "
        "may pick the best candidate, combine insights, or override them with your own "
        "answer if all are wrong. Output the final grid between OUTPUT: and END_OUTPUT markers."
    )
    usr_p = (
        f"{task.format_prompt()}\n\n"
        f"=== {len(candidates)} PROPOSER CANDIDATES ===\n{cand_block}\n\n"
        f"=== YOUR FINAL ANSWER ==="
    )
    use_thinking = client.model_key == "qwen-17b"
    try:
        resp = client.generate(
            sys_p, usr_p,
            temperature=0.5,
            max_tokens=MAX_TOKENS * 2,
            thinking=use_thinking,
        )
        return {
            "content": resp.content,
            "correct": task.check(resp.content),
            "tokens": resp.input_tokens + resp.output_tokens,
            "cost": resp.cost_usd,
        }
    except Exception as e:
        return {"content": "", "correct": False, "tokens": 0, "cost": 0, "error": str(e)[:80]}


# ════════════════════════════════════════════════════════════════════════
# ARCHITECTURE: LEAVE-ONE-OUT VERIFICATION
# ════════════════════════════════════════════════════════════════════════

def make_loo_task(task: ARCTask, hold_out_idx: int) -> ARCTask:
    """Create a leave-one-out variant: hide training[hold_out_idx], use it as test."""
    train_pairs = list(task.train_pairs)
    held = train_pairs.pop(hold_out_idx)
    return ARCTask(
        id=f"{task.id}_loo{hold_out_idx}",
        train_pairs=train_pairs,
        test_input=held[0],
        test_output=held[1],
    )


def score_proposer_on_loo(client: LLMClient, task: ARCTask, framing: str = None) -> tuple[int, int, dict]:
    """Score a proposer using leave-one-out on training examples.

    Returns (n_correct_loo, n_total_loo, test_attempt_dict).
    Each LOO call hides one training example and asks the model to predict it,
    then we get the test prediction in a separate call.
    """
    n_train = len(task.train_pairs)
    if n_train < 2:
        # Can't do LOO with only 1 training example
        test_attempt = call_arc(client, task, framing=framing)
        return 1 if test_attempt["correct"] else 0, 1, test_attempt

    # Build LOO calls
    loo_jobs = []
    for i in range(n_train):
        loo_task = make_loo_task(task, i)
        loo_jobs.append(loo_task)

    # Run LOO calls in parallel within this proposer
    n_correct_loo = 0
    with ThreadPoolExecutor(max_workers=min(len(loo_jobs), 10)) as pool:
        futures = [pool.submit(call_arc, client, lt, framing=framing) for lt in loo_jobs]
        for f in as_completed(futures):
            r = f.result()
            if r["correct"]:
                n_correct_loo += 1

    # Run the test attempt
    test_attempt = call_arc(client, task, framing=framing)
    return n_correct_loo, n_train, test_attempt


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

FRAMINGS = [
    "Focus on color patterns and how colors map between input and output.",
    "Focus on shapes and how they move, rotate, or transform.",
    "Focus on counting elements and how counts change between input and output.",
    "Focus on symmetry, mirroring, and reflection patterns.",
    "Focus on connectivity and how regions of same color are grouped.",
    "Focus on the bounding boxes of non-zero elements.",
    "Focus on how each row or column transforms independently.",
    "Focus on noise reduction or completing partial patterns.",
]


def exp_baseline(name: str, client: LLMClient, tasks: list[ARCTask]) -> dict:
    """Single-model single-shot baseline."""
    print(f"\n--- BASELINE: {name} ---", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(call_arc, client, t) for t in tasks]
        for f in as_completed(futures):
            records.append(f.result())
    k = sum(r["correct"] for r in records)
    n = len(records)
    cost = sum(r["cost"] for r in records)
    print(f"  {name}: {k}/{n} ({k/n:.1%}) cost ${cost:.4f}", flush=True)
    return {"name": name, "k": k, "n": n, "rate": k / n, "cost": cost, "type": "baseline"}


def exp_oracle_best_of_n(name: str, client: LLMClient, tasks: list[ARCTask], n_samples: int) -> dict:
    """Oracle best-of-N: count problems where AT LEAST ONE of N samples is correct."""
    print(f"\n--- ORACLE BEST-OF-{n_samples}: {name} ---", flush=True)
    per_task_correct = []
    total_cost = 0
    jobs = []
    for t in tasks:
        for i in range(n_samples):
            framing = FRAMINGS[i % len(FRAMINGS)]
            jobs.append((t, framing))

    results_by_task = defaultdict(list)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(call_arc, client, t, framing=fr): t.id for t, fr in jobs}
        for f in as_completed(futures):
            tid = futures[f]
            r = f.result()
            results_by_task[tid].append(r)
            total_cost += r["cost"]

    for t in tasks:
        recs = results_by_task[t.id]
        per_task_correct.append(any(r["correct"] for r in recs))

    k = sum(per_task_correct)
    print(f"  {name} best-of-{n_samples}: {k}/{len(tasks)} ({k/len(tasks):.1%}) cost ${total_cost:.4f}", flush=True)
    return {
        "name": f"{name}_best_of_{n_samples}",
        "k": k, "n": len(tasks), "rate": k / len(tasks),
        "cost": total_cost, "type": "oracle",
    }


def exp_synth_over_n(name: str, prop_clients: list[LLMClient], synth_client: LLMClient,
                      tasks: list[ARCTask]) -> dict:
    """N proposers (one per prop_clients entry) + 1 synthesizer.

    Each proposer uses a different framing.
    """
    n_props = len(prop_clients)
    print(f"\n--- SYNTH OVER N={n_props}: {name} ---", flush=True)
    synth_results = []
    total_cost = 0
    total_tokens = 0

    def run_one_episode(task: ARCTask):
        # Generate N proposals in parallel
        with ThreadPoolExecutor(max_workers=n_props) as inner_pool:
            futures = [
                inner_pool.submit(call_arc, prop_clients[i], task, framing=FRAMINGS[i % len(FRAMINGS)])
                for i in range(n_props)
            ]
            proposals = [f.result() for f in futures]
        # Synthesize
        synth = synthesize_over_candidates(synth_client, task, proposals)
        prop_cost = sum(p["cost"] for p in proposals)
        return {
            "task_id": task.id,
            "n_proposers_correct": sum(1 for p in proposals if p["correct"]),
            "synth_correct": synth["correct"],
            "prop_cost": prop_cost,
            "synth_cost": synth["cost"],
            "total_cost": prop_cost + synth["cost"],
        }

    # Run episodes in parallel — use a smaller pool here because each episode uses inner parallelism
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(run_one_episode, t) for t in tasks]
        completed = 0
        for f in as_completed(futures):
            r = f.result()
            synth_results.append(r)
            total_cost += r["total_cost"]
            completed += 1
            if completed % 5 == 0:
                k = sum(r["synth_correct"] for r in synth_results)
                print(f"  [{completed}/{len(tasks)}] {k}/{completed} synth correct, "
                      f"any-prop={sum(1 for r in synth_results if r['n_proposers_correct'] > 0)}/{completed}, "
                      f"cost ${total_cost:.4f}", flush=True)

    k = sum(r["synth_correct"] for r in synth_results)
    n = len(synth_results)
    any_prop_k = sum(1 for r in synth_results if r["n_proposers_correct"] > 0)
    avg_prop_correct = sum(r["n_proposers_correct"] for r in synth_results) / n

    print(f"  {name}: synth={k}/{n} ({k/n:.1%}) | any-proposer={any_prop_k}/{n} ({any_prop_k/n:.1%}) | "
          f"avg props correct {avg_prop_correct:.2f}/{n_props} | cost ${total_cost:.4f}", flush=True)
    return {
        "name": name,
        "k": k, "n": n, "rate": k / n,
        "any_proposer_rate": any_prop_k / n,
        "avg_proposers_correct": avg_prop_correct,
        "cost": total_cost,
        "cost_per_episode": total_cost / n,
        "type": "synth",
        "records": synth_results,
    }


def exp_loo_verification(name: str, client: LLMClient, tasks: list[ARCTask],
                          n_proposers: int) -> dict:
    """N proposers, each scored by leave-one-out on training examples.

    Final answer: the test prediction from the proposer with highest LOO score.
    Ties broken by majority vote among the top-scoring proposers.
    """
    print(f"\n--- LOO VERIFICATION N={n_proposers}: {name} ---", flush=True)

    def run_one_task(task: ARCTask):
        # Run N proposers, each with LOO scoring
        proposer_results = []
        with ThreadPoolExecutor(max_workers=n_proposers) as pool:
            futures = []
            for i in range(n_proposers):
                framing = FRAMINGS[i % len(FRAMINGS)]
                futures.append(pool.submit(score_proposer_on_loo, client, task, framing))
            for f in as_completed(futures):
                n_correct, n_total, test_attempt = f.result()
                proposer_results.append({
                    "loo_score": n_correct / max(n_total, 1),
                    "n_loo_correct": n_correct,
                    "n_loo_total": n_total,
                    "test_attempt": test_attempt,
                })
        # Find max LOO score
        max_score = max(p["loo_score"] for p in proposer_results)
        top_proposers = [p for p in proposer_results if p["loo_score"] == max_score]

        # Majority vote among top proposers
        votes = Counter()
        for p in top_proposers:
            c = parse_canonical(p["test_attempt"]["content"])
            if c:
                votes[c] += 1
        if votes:
            top_canonical = votes.most_common(1)[0][0]
            final_correct = check_canonical(top_canonical, task)
        else:
            final_correct = False

        prop_cost = sum(p["test_attempt"]["cost"] + sum(  # LOO calls counted separately not exposed; rough estimate
            0 for _ in range(p["n_loo_total"])
        ) for p in proposer_results)
        # Simpler: track only the test attempt cost; LOO costs are aggregated outside
        return {
            "task_id": task.id,
            "max_loo_score": max_score,
            "n_top_proposers": len(top_proposers),
            "final_correct": final_correct,
            "any_proposer_correct": any(p["test_attempt"]["correct"] for p in proposer_results),
        }

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one_task, t) for t in tasks]
        completed = 0
        for f in as_completed(futures):
            try:
                r = f.result()
                records.append(r)
                completed += 1
                if completed % 5 == 0:
                    k = sum(r["final_correct"] for r in records)
                    cost = client.tracker.total_cost
                    print(f"  [{completed}/{len(tasks)}] {k}/{completed} correct, "
                          f"any-prop={sum(1 for r in records if r['any_proposer_correct'])}/{completed}, "
                          f"cost ${cost:.4f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)

    k = sum(r["final_correct"] for r in records)
    n = len(records)
    any_correct_k = sum(1 for r in records if r["any_proposer_correct"])
    print(f"  {name}: LOO-pick={k}/{n} ({k/n:.1%}) | any-prop={any_correct_k}/{n} ({any_correct_k/n:.1%}) | "
          f"cost ${client.tracker.total_cost:.4f}", flush=True)
    return {
        "name": name,
        "k": k, "n": n, "rate": k / n,
        "any_proposer_rate": any_correct_k / n,
        "cost": client.tracker.total_cost,
        "type": "loo",
        "records": records,
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main(dry_run: bool = False, only: list[str] | None = None):
    print("=" * 70)
    print("ARC CRACKING SUITE — multiple architectures targeting ≥30%")
    print("=" * 70)
    print(f"  Tasks: {N_PROBLEMS}")
    print(f"  Workers: {WORKERS} (aggressive parallelism)")

    if dry_run:
        # Rough cost estimates
        n = N_PROBLEMS
        baselines = 3 * n  # 3 models × 30 tasks
        oracles = 3 * n * 16  # 3 × 30 × 16 samples
        synth_a = n * 17  # 16 props + 1 synth × 30 tasks
        synth_b = n * 17  # similar
        synth_c = n * 17  # similar
        loo_c1 = n * 16 * 4  # 16 proposers × ~4 calls per LOO
        loo_c2 = n * 16 * 4
        total_calls = baselines + oracles + synth_a + synth_b + synth_c + loo_c1 + loo_c2
        # ARC calls cost ~$0.001-0.01 depending on model
        avg_call_cost = 0.003  # weighted average
        est = total_calls * avg_call_cost
        print(f"\n[DRY RUN] ~{total_calls} calls, est ${est:.2f}")
        print(f"[DRY RUN] (rough estimate; exact depends on token usage)")
        return

    print(f"\nLoading ARC-AGI...")
    tasks = load_arc_tasks(n=N_PROBLEMS, seed=42)
    print(f"  {len(tasks)} tasks loaded")

    clients = {
        "qwen-7b": LLMClient("qwen-7b"),
        "qwen-22b": LLMClient("qwen-22b"),
        "qwen-17b": LLMClient("qwen-17b"),
    }

    output_dir = RESULTS_DIR / "arc_crack"
    output_dir.mkdir(exist_ok=True)
    t_start = time.time()
    all_results = {}

    def should_run(exp_name: str) -> bool:
        return only is None or exp_name in only

    # ── BASELINES ──────────────────────────────────────────────────────
    if should_run("baselines"):
        for m in ["qwen-7b", "qwen-22b", "qwen-17b"]:
            r = exp_baseline(f"single-{m}", clients[m], tasks)
            all_results[r["name"]] = r
        # Save partial
        json.dump({"phase": "baselines", "results": all_results}, open(output_dir / "partial_1_baselines.json", "w"), indent=2, default=str)

    # ── ORACLE BEST-OF-N (theoretical ceiling for any best-of-N strategy) ──
    if should_run("oracles"):
        for m in ["qwen-7b", "qwen-22b"]:
            r = exp_oracle_best_of_n(m, clients[m], tasks, n_samples=16)
            all_results[r["name"]] = r
        json.dump({"phase": "oracles", "results": all_results}, open(output_dir / "partial_2_oracles.json", "w"), indent=2, default=str)

    # ── SYNTH OVER N ARCHITECTURES ─────────────────────────────────────
    if should_run("synth"):
        # EXP-A1: 16×qwen-7b + qwen-17b synth
        prop_clients = [clients["qwen-7b"]] * 16
        r = exp_synth_over_n("EXP-A1__16x7b_synth17b", prop_clients, clients["qwen-17b"], tasks)
        all_results[r["name"]] = r
        json.dump({"phase": "synth", "results": all_results}, open(output_dir / "partial_3_synth.json", "w"), indent=2, default=str)

        # EXP-A2: 16×qwen-7b + qwen-22b synth (cheaper synth)
        r = exp_synth_over_n("EXP-A2__16x7b_synth22b", prop_clients, clients["qwen-22b"], tasks)
        all_results[r["name"]] = r
        json.dump({"phase": "synth", "results": all_results}, open(output_dir / "partial_3_synth.json", "w"), indent=2, default=str)

        # EXP-B1: heterogeneous 8×7b + 4×22b + 2×17b + 17b synth
        prop_clients_het = (
            [clients["qwen-7b"]] * 8
            + [clients["qwen-22b"]] * 4
            + [clients["qwen-17b"]] * 2
        )
        r = exp_synth_over_n("EXP-B1__8x7b_4x22b_2x17b_synth17b", prop_clients_het, clients["qwen-17b"], tasks)
        all_results[r["name"]] = r
        json.dump({"phase": "synth", "results": all_results}, open(output_dir / "partial_3_synth.json", "w"), indent=2, default=str)

    # ── SUMMARY ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_cost = sum(c.tracker.total_cost for c in clients.values())

    print(f"\n{'='*70}")
    print(f"ARC CRACKING SUITE — RESULTS")
    print(f"{'='*70}")

    print(f"\n  {'Name':45s} {'Type':10s} {'Rate':>8s} {'Cost':>10s}")
    print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*10}")

    sorted_results = sorted(all_results.values(), key=lambda x: -x["rate"])
    for r in sorted_results:
        rate_str = f"{r['rate']:.1%}"
        cost_str = f"${r['cost']:.4f}"
        marker = "★" if r["rate"] >= 0.30 else " "
        print(f"  {marker} {r['name']:43s} {r['type']:10s} {rate_str:>8s} {cost_str:>10s}")

    print(f"\n  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Identify the headline result
    games = [r for r in sorted_results if r["type"] in ("synth", "loo")]
    frontier = all_results.get("single-qwen-17b")
    if games and frontier:
        best = games[0]
        delta = best["rate"] - frontier["rate"]
        print(f"\n--- HEADLINE ---")
        print(f"  Best game: {best['name']}")
        print(f"  Rate: {best['rate']:.1%} (frontier baseline: {frontier['rate']:.1%}, Δ={delta:+.1%})")
        if best["rate"] >= 0.30:
            print(f"  ★★★ CRACKED 30% TARGET ★★★")
        elif delta > 0:
            print(f"  ✓ Beats frontier by {delta:+.1%}")
        else:
            print(f"  ✗ Does not beat frontier")

    json.dump({
        "experiment": "arc_crack",
        "n_problems": len(tasks),
        "results": all_results,
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only = sys.argv[idx + 1].split(",")
    main(dry_run="--dry-run" in sys.argv, only=only)
