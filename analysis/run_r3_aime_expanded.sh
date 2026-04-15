#!/usr/bin/env bash
# R3: Expand AIME from 30 → 60 problems (2024 + 2025)
# Cost: ~$0.30 (AIME 2025 only — 2024 already cached)
# Time: ~1h
set -e
cd /Users/mouzouni/Documents/Dev/verifier-is-all-you-need
PYTHON="/Users/mouzouni/Documents/Dev/agent-security-paper/.venv/bin/python"

echo "=== R3: AIME 2025 (30 new problems, K=16) ==="
$PYTHON -u -m alpha_program.exp_e8_aime \
    --year 2025 --n -1 --k 16 --workers 16 \
    --output r3_aime2025_k16.json

echo
echo "=== R3 done ==="
