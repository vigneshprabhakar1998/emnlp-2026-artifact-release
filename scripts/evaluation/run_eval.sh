#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — Reproduce the evaluation results from the paper
# =============================================================================
#
# This script evaluates the Distilled-1B instruction-following reranker
# on the validation benchmark (9,861 queries) and MAIR OOD subsets
# (869 queries).
#
# Prerequisites:
#   1. Create a virtual environment and install dependencies:
#        python -m venv .venv
#        source .venv/bin/activate
#        pip install -r requirements.txt
#
#   2. Set your HuggingFace token (required to download the model):
#        export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#      Generate one at: https://huggingface.co/settings/tokens
#
# Hardware:
#   Any GPU with >=4GB VRAM (V100, A100, H100, etc.)
#   Runtime: ~20 minutes on a single GPU
#
# Usage:
#   bash scripts/evaluation/run_eval.sh
#
# =============================================================================

set -euo pipefail

# ---- Environment variables ----
# HuggingFace token: required to download model weights.
# Set this before running, or uncomment and paste your token below:
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set."
    echo "  The model download may fail if the repository is gated."
    echo "  Generate a token at: https://huggingface.co/settings/tokens"
    echo "  Then: export HF_TOKEN=\"hf_...\""
    echo ""
fi

# Optional: set CUDA device (default: GPU 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional: redirect HF cache to a directory with sufficient space
# export HF_HOME="/path/to/large/disk/.hf_cache"

# ---- Configuration ----
MODEL_PATH="anonymousauthor01/instruction_following_reranker"

# Paths relative to repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
VAL_JSONL="${REPO_ROOT}/datasets/train-val-set/val.jsonl"
MAIR_ROOT="${REPO_ROOT}/datasets/MAIR_OOD"
OUT_DIR="${REPO_ROOT}/results"

mkdir -p "${OUT_DIR}"

# ---- Verify prerequisites ----
if [ ! -f "${VAL_JSONL}" ]; then
    echo "ERROR: val.jsonl not found at ${VAL_JSONL}"
    echo "  Make sure you are running from the repo root directory."
    exit 1
fi

if [ ! -d "${MAIR_ROOT}" ]; then
    echo "WARNING: MAIR_OOD directory not found at ${MAIR_ROOT}"
    echo "  Will evaluate on validation set only."
    MAIR_FLAG=""
else
    MAIR_FLAG="--mair_root ${MAIR_ROOT}"
fi

echo "============================================================"
echo "  Instruction-Following Reranker — Evaluation"
echo "============================================================"
echo ""
echo "  Model:    ${MODEL_PATH}"
echo "  Val set:  ${VAL_JSONL}"
echo "  MAIR:     ${MAIR_ROOT}"
echo "  Output:   ${OUT_DIR}"
echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

# ---- Run evaluation ----
python "${SCRIPT_DIR}/evaluate.py" \
  --model_path "${MODEL_PATH}" \
  --val_jsonl "${VAL_JSONL}" \
  ${MAIR_FLAG} \
  --out_dir "${OUT_DIR}" \
  --k 6 \
  --bf16 \
  --seed 42 \
  2>&1 | tee "${OUT_DIR}/eval.log"

echo ""
echo "============================================================"
echo "  Done. Results saved to ${OUT_DIR}/evaluation_results.json"
echo "============================================================"
