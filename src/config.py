"""Configuration for pivot-a experiments.

ALL experiments use the Qwen model family exclusively to avoid confounding
model architecture effects with mechanism effects. Model size is the key
experimental parameter.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Qwen model family (serverless on Together AI, verified 2026-04) ────
# Model size is the experimental parameter: small → medium → frontier
MODELS = {
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",          # 7B dense, small
    "qwen-22b": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", # 235B MoE / 22B active, medium
    "qwen-17b": "Qwen/Qwen3.5-397B-A17B",                  # 397B MoE / 17B active, frontier
    # Tier 1 model-family replication only
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek": "deepseek-ai/DeepSeek-V3.1",
    # AlphaProgram code-specialist proposers
    "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-coder-480b": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    # AlphaProgram frontier reasoner baselines
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1-distill-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwq-32b": "Qwen/QwQ-32B",
}

# Cost per million tokens (Together AI pricing, verified 2026-04)
COST_PER_M_INPUT = {
    "qwen-7b": 0.30,
    "qwen-22b": 0.20,
    "qwen-17b": 0.60,
    "llama-70b": 0.88,
    "deepseek": 0.60,
    "qwen-coder-32b": 0.80,
    "qwen3-coder-30b": 0.00,   # promotional / preview tier as of API list
    "qwen3-coder-480b": 2.00,
    "deepseek-r1": 3.00,
    "deepseek-r1-distill-70b": 2.00,
    "qwq-32b": 1.20,
}
COST_PER_M_OUTPUT = {
    "qwen-7b": 0.30,
    "qwen-22b": 0.60,
    "qwen-17b": 3.60,
    "llama-70b": 0.88,
    "deepseek": 1.70,
    "qwen-coder-32b": 0.80,
    "qwen3-coder-30b": 0.00,
    "qwen3-coder-480b": 2.00,
    "deepseek-r1": 7.00,
    "deepseek-r1-distill-70b": 2.00,
    "qwq-32b": 1.20,
}

# ── Tier definitions for escalation experiments ────────────────────────
TIERS = {
    1: {"propose": "qwen-7b", "critique": "qwen-7b", "verify": "qwen-7b", "insight": "qwen-7b"},
    2: {"propose": "qwen-22b", "critique": "qwen-22b", "verify": "qwen-7b", "insight": "qwen-22b"},
    3: {"propose": "qwen-17b", "critique": "qwen-17b", "verify": "qwen-22b", "insight": "qwen-17b"},
}

# Ordered by effective size (for sweep experiments)
MODEL_SIZES_ORDERED = ["qwen-7b", "qwen-22b", "qwen-17b"]

# ── API keys ───────────────────────────────────────────────────────────
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
