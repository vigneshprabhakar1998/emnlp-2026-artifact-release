# Stage 2: On-Policy GRPO Distillation

This directory contains the training code for **Stage 2** of the paper — the
on-policy GRPO distillation that produces the final **Distilled-1B** reranker
(the row **"A3: On-policy GRPO from Teacher (ours)"** in the OOD ablation table).

## What it does

A frozen **teacher** (a ZeRank-2-style CausalLM reranker, scored via the
"Yes"-token logit) supplies both the ranking-reward relevances and a soft
distillation target. A trainable **student** (Llama-Nemotron-Rerank-1B, a
sequence-classification cross-encoder) is optimized with:

```
loss = pg_loss
     + lambda_kl_distill * KL(student || teacher)
     - ent_coef * H(student)
```

- **`pg_loss`** — GRPO-style policy gradient. For each query, `group_size`
  slates are drawn from the student's scores via Plackett–Luce sampling, each
  rewarded by **nDCG@k** against the teacher-derived relevances, and the
  advantage is group-normalized `(r - mean) / (std + eps)`.
- **KL term** — pulls the student's softmax score distribution toward the
  teacher's softmax distribution.
- **Entropy term** — small entropy bonus for exploration.

## Quick start

```bash
bash scripts/training/launch.sh
```

This downloads the train/val split from HuggingFace (if not present locally),
loads the teacher and student, and trains for one epoch, periodically evaluating
student nDCG@6 against the teacher and saving the top-k checkpoints.

## Configuration

`launch.sh` uses the exact hyperparameters reported in the paper (group size 8,
nDCG@6 reward, KL weight 1.0, entropy coefficient 0.01, LR 2e-6, `rank` score
normalization). Every value is overridable via an environment variable, e.g.:

```bash
TEACHER_PATH=/path/to/stage1_teacher_ckpt \
STUDENT_ID=nvidia/llama-nemotron-rerank-1b-v2 \
GROUP_SIZE=8 LR=2e-6 OUT_DIR=results/distill_stage2 \
  bash scripts/training/launch.sh
```

Key arguments (see `on_policy_rl.py --help` for the full list):

| Argument | Default | Meaning |
|----------|---------|---------|
| `--teacher_path` | `zeroentropy/zerank-2` | Frozen teacher (local Stage 1 checkpoint or HF id) |
| `--student_id` | `nvidia/llama-nemotron-rerank-1b-v2` | Trainable student backbone |
| `--group_size` | `8` | GRPO slates sampled per query |
| `--k` | `6` | nDCG@k reward cutoff |
| `--pl_temperature` | `1.0` | Plackett–Luce sampling temperature |
| `--lambda_kl_distill` | `1.0` | Weight on the KL distillation term |
| `--ent_coef` | `0.01` | Entropy bonus coefficient |
| `--score_norm` | `rank` | Teacher-score → relevance normalization |
| `--lr` | `2e-6` | AdamW learning rate |

To reproduce the paper's teacher, use your Stage 1 checkpoint as
`--teacher_path`. The default (`zeroentropy/zerank-2`) makes the script runnable
out of the box against the public base reranker.

## Hardware

- **1× H100** (or 2× A100 80GB), bfloat16.
- ~4h50m for one epoch over the training split.

## Outputs

Written under `OUT_DIR` (default `results/distill_stage2/`):

- `checkpoints/ckpt_step*_valndcg*/` — top-k student checkpoints (by val nDCG@k).
- `best_checkpoints.json` — ranked list of retained checkpoints.
- `last_eval.json`, `final_metrics.json` — validation metrics.
- `train.log` — console log.
