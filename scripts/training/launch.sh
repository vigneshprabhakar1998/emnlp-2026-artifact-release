#!/usr/bin/env bash
# =============================================================================
# launch.sh — Stage 2: On-policy GRPO distillation (student training)
# =============================================================================
#
# Trains the Distilled-1B instruction-following reranker with on-policy GRPO
# distillation from a frozen 4B teacher, reproducing Stage 2 of the paper
# (row "A3: On-policy GRPO from Teacher" in the ablation table).
#
#   Teacher  : ZeRank-2-style CausalLM reranker (frozen), scored via the
#              "Yes"-token logit. Provides the reference relevances (nDCG@k
#              reward target) and the soft KL distillation target.
#   Student  : Llama-Nemotron-Rerank-1B (sequence-classification cross-encoder),
#              trainable. Optimized with:
#                loss = pg_loss
#                     + lambda_kl_distill * KL(student || teacher)
#                     - ent_coef * H(student)
#
# Datasets: the same train/val split used for evaluation is downloaded
# automatically from HuggingFace if not present locally:
#   - anonymousauthor01/emnlp-2026-ifr-train-val-set  (train.jsonl, val.jsonl)
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
#   1x H100 (or 2x A100 80GB). ~4h50m for 1 epoch over the training split.
#
# Usage:
#   bash scripts/training/launch.sh
#
# Common overrides (environment variables):
#   TEACHER_PATH=... STUDENT_ID=... OUT_DIR=... GROUP_SIZE=8 LR=2e-6 \
#     bash scripts/training/launch.sh
#
# =============================================================================

set -euo pipefail

# ---- Environment variables ----
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN / HUGGINGFACE_HUB_TOKEN is not set."
    echo "  Downloads will work only if the model and dataset repositories are public."
    echo "  Generate a token at: https://huggingface.co/settings/tokens"
    echo ""
fi

# Optional: set CUDA device (default: GPU 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional: redirect HF cache to a directory with sufficient space
# export HF_HOME="/path/to/large/disk/.hf_cache"

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/distill_stage2}"

# ---- Model configuration ----
# Teacher: a local path to the Stage 1 teacher checkpoint, OR a HuggingFace id.
# Replace with your Stage 1 teacher checkpoint. Defaults to the public base
# ZeRank-2 reranker so the script is runnable out of the box.
TEACHER_PATH="${TEACHER_PATH:-zeroentropy/zerank-2}"
STUDENT_ID="${STUDENT_ID:-nvidia/llama-nemotron-rerank-1b-v2}"

# ---- Dataset (train/val split) ----
VAL_DATASET_ID="${VAL_DATASET_ID:-anonymousauthor01/emnlp-2026-ifr-train-val-set}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/train-val-set/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/train-val-set/val.jsonl}"

# Auto-download the train/val split from HuggingFace if not already present.
if [ ! -f "${TRAIN_JSONL}" ] || [ ! -f "${VAL_JSONL}" ]; then
    echo "[data] Train/val split not found locally; downloading from HuggingFace..."
    huggingface-cli download "${VAL_DATASET_ID}" \
        --repo-type dataset \
        --local-dir "${DATA_DIR}/train-val-set"
fi

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "  Instruction-Following Reranker — Stage 2 Distillation"
echo "============================================================"
echo ""
echo "  Teacher:  ${TEACHER_PATH}"
echo "  Student:  ${STUDENT_ID}"
echo "  Train:    ${TRAIN_JSONL}"
echo "  Val:      ${VAL_JSONL}"
echo "  Output:   ${OUT_DIR}"
echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

# ---- Hyperparameters (paper configuration, A3) ----
python "${SCRIPT_DIR}/on_policy_rl.py" \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --teacher_path "${TEACHER_PATH}" \
  --student_id "${STUDENT_ID}" \
  --out_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-1}" \
  --max_len "${MAX_LEN:-512}" \
  --k "${K:-6}" \
  --group_size "${GROUP_SIZE:-8}" \
  --pl_temperature "${PL_TEMPERATURE:-1.0}" \
  --teacher_temp "${TEACHER_TEMP:-1.0}" \
  --student_temp "${STUDENT_TEMP:-1.0}" \
  --score_norm "${SCORE_NORM:-rank}" \
  --lambda_kl_distill "${LAMBDA_KL_DISTILL:-1.0}" \
  --ent_coef "${ENT_COEF:-0.01}" \
  --batch_size_rows "${BATCH_SIZE_ROWS:-32}" \
  --grad_accum "${GRAD_ACCUM:-2}" \
  --lr "${LR:-2e-6}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --clip_grad "${CLIP_GRAD:-1.0}" \
  --bf16 \
  --score_chunk_size "${SCORE_CHUNK_SIZE:-32}" \
  --seed "${SEED:-0}" \
  --log_every "${LOG_EVERY:-10}" \
  --eval_every "${EVAL_EVERY:-138}" \
  --save_every "${SAVE_EVERY:-138}" \
  --save_top_k "${SAVE_TOP_K:-3}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "============================================================"
echo "  Done. Checkpoints + metrics saved to ${OUT_DIR}"
echo "============================================================"
