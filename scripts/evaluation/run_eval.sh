#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — Reproduce the evaluation results from the paper
# =============================================================================
#
# This script evaluates the Distilled-1B instruction-following reranker
# on the validation benchmark (9,861 queries) and MAIR OOD subsets
# (869 queries).
#
# Datasets are downloaded automatically from HuggingFace:
#   - anonymousauthor01/emnlp-2026-ifr-train-val-set
#   - anonymousauthor01/emnlp-2026-ifr-mair-ood
#
# Prerequisites:
#   1. Create a virtual environment and install dependencies:
#        python -m venv .venv
#        source .venv/bin/activate
#        pip install -r requirements.txt
#
#   2. Set your HuggingFace token if the model/datasets are private or gated:
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
# HuggingFace token: required if the model or dataset repositories are gated.
# Set this before running, or uncomment and paste your token below:
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN / HUGGINGFACE_HUB_TOKEN is not set."
    echo "  Downloads will work only if the model and dataset repositories are public."
    echo "  Generate a token at: https://huggingface.co/settings/tokens"
    echo "  Then: export HF_TOKEN=\"hf_...\""
    echo ""
fi

# Optional: set CUDA device (default: GPU 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional: redirect HF cache to a directory with sufficient space
# export HF_HOME="/path/to/large/disk/.hf_cache"

# ---- Configuration ----
MODEL_PATH="${MODEL_PATH:-anonymousauthor01/instruction_following_reranker}"
VAL_DATASET_ID="${VAL_DATASET_ID:-anonymousauthor01/emnlp-2026-ifr-train-val-set}"
MAIR_DATASET_ID="${MAIR_DATASET_ID:-anonymousauthor01/emnlp-2026-ifr-mair-ood}"

# Paths relative to repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results}"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

echo "============================================================"
echo "  Instruction-Following Reranker — Evaluation"
echo "============================================================"
echo ""
echo "  Model:       ${MODEL_PATH}"
echo "  Val HF repo: ${VAL_DATASET_ID}"
echo "  MAIR HF repo:${MAIR_DATASET_ID}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Output:      ${OUT_DIR}"
echo "  GPU:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

# ---- Run evaluation ----
python "${SCRIPT_DIR}/evaluate.py" \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --val_dataset_id "${VAL_DATASET_ID}" \
  --mair_dataset_id "${MAIR_DATASET_ID}" \
  --out_dir "${OUT_DIR}" \
  --k 6 \
  --bf16 \
  --seed 42 \
  2>&1 | tee "${OUT_DIR}/eval.log"

echo ""
echo "============================================================"
echo "  Done. Results saved to ${OUT_DIR}/evaluation_results.json"
echo "============================================================"

