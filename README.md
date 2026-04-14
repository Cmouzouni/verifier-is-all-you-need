# The Verifier Is All You Need

**Six Architectural Interventions That Don't Matter and the Verification Boundary That Does**

*Charafeddine Mouzouni*
OPIT — Open Institute of Technology, and Cohorte AI, Paris, France.

---

## Abstract

We test six architectural interventions for multi-agent LLM reasoning — game-theoretic allocation, critic-driven refinement, increased sampling budget, code-specialist proposers, heterogeneous multi-model ensembles, and evolutionary inter-round information flow — and find that **none of them improves over the simplest possible baseline**: sample K programs independently from a single model, execute each in a sandbox, return the first that passes verification. The architecture does not matter. The verifier is all you need.

This finding emerges from a systematic empirical campaign (~7,000 LLM-agent episodes, ~$40 in API cost) spanning five benchmarks:

| Benchmark | Result | vs Frontier | Same-task baseline |
|---|---|---|---|
| **HumanEval** | 97.0% pass@16 | +4pp over o1's pass@1 | — |
| **AIME 2024** | 76.7% (23/30) | 2.2× cheaper than same-task o4-mini (80%) | **o4-mini: 24/30 = 80%** |
| **ARC-AGI-1** | 18.2% (73/400) | Lowest cost-per-correct ($0.16) | DeepSeek-V3.1: 2.8% |
| **ARC-AGI-2** | 0% (0/100) | Null — DSL can't express abstractions | — |

Six architectural mechanisms tested, all null:

| # | Mechanism | Result |
|---|---|---|
| 1 | MFG game-theoretic allocation | Ties random (p>0.4) |
| 2 | Critic-driven failure feedback | 40% vs 43% cold baseline |
| 3 | Sampling budget (K sweep) | K=8 ≈ K=64 |
| 4 | Code-specialist proposer | Worse than generalist |
| 5 | 3-model heterogeneous union | 16.5% vs 17.4% single model |
| 6 | Evolutionary inter-round info flow | 43.3% = 43.3% cold K=8 |

## Repository Structure

```
verifier-is-all-you-need/
├── paper/
│   └── main.tex                    # Full paper (36 pages, amsart format)
├── src/
│   ├── client.py                   # LLM API client (Together AI)
│   ├── config.py                   # Model registry & pricing
│   ├── congestion.py               # MFG congestion mechanism
│   ├── diversity.py                # Diversity metrics
│   ├── runner.py                   # Episode runner
│   └── schemas.py                  # Data schemas
├── tasks/
│   ├── arc_tasks.py                # ARC-AGI-1/2 loader (HuggingFace)
│   ├── aime_tasks.py               # AIME 2024 loader
│   ├── gsm8k_tasks.py              # GSM8K loader
│   └── phase_a_tasks.py            # Phase A math battery
├── alpha_program/
│   ├── dsl.py                      # ARC grid-primitive library (30 primitives)
│   ├── verifier.py                 # Sandboxed Python executor for ARC
│   ├── aime_verifier.py            # Sandboxed executor for AIME (sympy)
│   ├── run_validation.py           # Unified validation runner
│   ├── run_frontier_baseline.py    # Frontier model baseline runner
│   ├── exp_e2_verifier_loop.py     # E2: verifier-in-loop best-of-N
│   ├── exp_e3_critic_loop.py       # E3: critic loop (null result)
│   ├── exp_e8_aime.py              # E8: AIME with code execution
│   ├── exp_e9_humaneval.py         # E9: HumanEval with unit tests
│   ├── exp_dirA_evolutionary.py    # Dir A: evolutionary refinement (null)
│   ├── exp_dirB_heterogeneous_union.py  # Dir B: model diversity (null)
│   ├── exp_aime_o1mini_baseline.py # Same-task o4-mini AIME baseline
│   ├── analyze_validation.py       # Results analyzer with Wilson CIs
│   └── plot_pareto.py              # Cost-correctness Pareto plot
├── experiments/
│   └── ...                         # Act I experiments (factorial sweep, ablations)
├── results/
│   └── alpha_program/
│       ├── e7_arc1_eval_n400_k16.json    # ARC-AGI-1 full 400-task eval
│       ├── e8_aime2024_k16_v2.json       # AIME 2024 all 30 problems
│       ├── e9_humaneval_k16.json         # HumanEval all 164 problems
│       ├── aime_o4mini_baseline.json     # Same-task o4-mini baseline
│       ├── v2a_deepseek_v31_eval400_fixed.json  # Same-task DeepSeek baseline
│       ├── v3_arc2_pilot_qwen22b_k16.json       # ARC-AGI-2 null result
│       ├── e3_critic_loop_training30.json        # Critic null result
│       ├── dirA_evolutionary_training30_v2.json  # Evolutionary null result
│       ├── dirB_llama70b_eval400_k8.json         # Model diversity null
│       ├── v5_ksweep_training30_k8.json          # K-sweep data
│       ├── v5_ksweep_training30_k64.json         # K-sweep data
│       └── ...                                    # All intermediate results
└── requirements.txt
```

## Reproducing the Results

### Setup

```bash
git clone https://github.com/Cmouzouni/verifier-is-all-you-need.git
cd verifier-is-all-you-need
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API keys
echo "TOGETHER_API_KEY=your_key_here" > .env
echo "OPENAI_API_KEY=your_key_here" >> .env  # only needed for o4-mini baseline
```

### Running the key experiments

**E8 — AIME 2024 (the crown jewel, ~$0.15, ~2 hours):**
```bash
python -m alpha_program.exp_e8_aime --year 2024 --k 16 --output my_aime_run.json
```

**E9 — HumanEval (97% pass@16, ~$0.50, ~20 minutes):**
```bash
python -m alpha_program.exp_e9_humaneval --k 16 --output my_humaneval_run.json
```

**ARC-AGI-1 full eval (18.2%, ~$12, ~24 hours on Together AI):**
```bash
python -m alpha_program.run_validation \
    --model qwen-22b --dataset arc1 --split evaluation \
    --n 400 --k 16 --workers 16 \
    --output my_arc_eval.json
```

**Same-task o4-mini AIME baseline (~$0.35, ~5 minutes):**
```bash
python -m alpha_program.exp_aime_o1mini_baseline \
    --model o4-mini --output my_o4mini_baseline.json
```

**Six-null falsification battery (~$5 total):**
```bash
# E3: critic loop
python -m alpha_program.exp_e3_critic_loop --n 30 --output my_e3.json

# Dir A: evolutionary refinement
python -m alpha_program.exp_dirA_evolutionary --n 30 --k-per-round 8 --rounds 3 --output my_dirA.json

# Dir B: heterogeneous union
python -m alpha_program.run_validation --model deepseek --dataset arc1 --split evaluation --n 400 --k 8 --output my_dirB_deepseek.json
python -m alpha_program.run_validation --model llama-70b --dataset arc1 --split evaluation --n 400 --k 8 --output my_dirB_llama.json
python -m alpha_program.exp_dirB_heterogeneous_union --analyze
```

### Pre-computed results

All experimental results are included in `results/` as JSON files. To analyze without re-running:

```bash
python -m alpha_program.analyze_validation
```

## Key Findings

1. **The verifiability boundary** — LLM synthesis helps +14-29pp on verifiable math tasks, actively hurts -3 to -23pp on ARC visual grids. The boundary is mechanistic: the LLM aggregator helps when it can verify candidates by re-reading them and hurts when verification requires re-deriving the problem.

2. **Symbolic verification dissolves the boundary** — Replacing the LLM aggregator with sandboxed Python execution achieves frontier-competitive accuracy on HumanEval (97%), AIME (77%), and ARC-AGI-1 (18%) at 2-55× lower cost per correct answer.

3. **Six architectural mechanisms are null** — Game-theoretic allocation, critic feedback, sampling budget, code-specialist proposers, model diversity, and evolutionary refinement all fail to improve over the trivial baseline (independent sampling + execution + vote).

4. **The outer boundary is DSL coverage** — ARC-AGI-2 produces 0/100 because the Python DSL can't express the required abstractions. Frontier models also score <5%.

## Citation

```bibtex
@article{mouzouni2026verifier,
  title={The Verifier Is All You Need: Six Architectural Interventions That Don't Matter and the Verification Boundary That Does},
  author={Mouzouni, Charafeddine},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Total experimental cost: ~$41 in API fees (Together AI + OpenAI). The complete dataset of ~7,000 LLM-agent episodes is included in this repository.
