"""Proposer-verifier-vote architecture for hard reasoning benchmarks.

Implements the symbolic-verification approach described in the paper:
cheap LLM proposers generate executable programs, a sandboxed Python
interpreter verifies each candidate, and majority vote selects the
final answer.

Modules:
  dsl.py             -- ARC grid-primitive library (30 primitives)
  verifier.py        -- Sandboxed executor for ARC transformation programs
  aime_verifier.py   -- Sandboxed executor for AIME solver functions (sympy)
  run_validation.py  -- Unified experiment runner
  analyze_validation.py -- Results analyzer with Wilson 95% CIs
"""
