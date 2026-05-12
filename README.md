# On-Policy Distillation Meets Off-Policy GRPO: Training Sparse Instruction-Following Rerankers that Beat Their Teachers

**Anonymous EMNLP 2025 Submission**

---

## Overview

This repository provides the evaluation code, dataset download hooks, and model checkpoint necessary to reproduce the main results reported in the paper. The final **Distilled-1B** reranker achieves:

| Benchmark | nDCG@6 | MRR@6 | 95% CI (nDCG) |
|-----------|--------|-------|---------------|
| Validation (9,861 queries) | 0.7624 | 0.7475 | [0.755, 0.770] |
| MAIR OOD (869 queries) | 0.7670 | 0.8289 | [0.745, 0.789] |

## Quick Start

### 1. Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your HuggingFace token

The model checkpoint and datasets are hosted on HuggingFace. If any repository is private or gated, set a token before running:

```bash
# Generate a token at: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Run evaluation

```bash
bash scripts/evaluation/run_eval.sh
```

This will:
- Download the model from HuggingFace (~2.5 GB)
- Auto-download the validation train/val set from HuggingFace if not present locally
- Auto-download the MAIR OOD dataset from HuggingFace if not present locally
- Evaluate on the validation benchmark (8 datasets, 9,861 queries)
- Evaluate on MAIR OOD subsets (11 subsets, 869 queries)
- Report nDCG@6, MRR@6 with bootstrap 95% CIs
- Save results to `results/evaluation_results.json`

**Runtime:** ~20 minutes on a single GPU.

### 4. Custom evaluation

Default HF-backed evaluation:

```bash
python scripts/evaluation/evaluate.py \
  --model_path anonymousauthor01/instruction_following_reranker \
  --data_dir datasets \
  --out_dir results \
  --bf16
```

Explicit HF dataset repos:

```bash
python scripts/evaluation/evaluate.py \
  --model_path anonymousauthor01/instruction_following_reranker \
  --val_dataset_id anonymousauthor01/emnlp-2026-ifr-train-val-set \
  --mair_dataset_id anonymousauthor01/emnlp-2026-ifr-mair-ood \
  --data_dir datasets \
  --out_dir results \
  --bf16
```

Offline/local evaluation after manual download:

```bash
python scripts/evaluation/evaluate.py \
  --model_path /path/to/instruction_following_reranker \
  --val_jsonl datasets/train-val-set/val.jsonl \
  --mair_root datasets/MAIR_OOD \
  --out_dir results \
  --bf16
```

Additional options:
- `--device cuda:1` — specify GPU
- `--k 10` — change the cutoff (default: 6)
- `--seed 42` — random seed for bootstrap CIs
- `--skip_mair` — evaluate only the validation set
- `--force_download` — refresh HuggingFace dataset snapshots

## Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies (pinned versions)
├── scripts/
│   └── evaluation/
│       ├── evaluate.py                    # Evaluation script
│       └── run_eval.sh                    # One-command launcher
├── datasets/                              # Created automatically if absent
│   ├── train-val-set/                     # HF snapshot: emnlp-2026-ifr-train-val-set
│   │   └── val.jsonl                      # Validation queries (9,861)
│   └── MAIR_OOD/                          # HF snapshot: emnlp-2026-ifr-mair-ood
│       ├── docs/                          # Document corpora per subset
│       │   ├── ArguAna/
│       │   ├── Core_2017/
│       │   └── ...
│       └── queries/                       # Queries + relevance labels per subset
│           ├── ArguAna/
│           ├── Core_2017/
│           └── ...
└── results/                               # Output directory created by run_eval.sh
    ├── eval.log                           # Console log
    └── evaluation_results.json            # Full metrics
```

## Model Checkpoint

The Distilled-1B checkpoint is hosted at:
**[https://huggingface.co/anonymousauthor01/instruction_following_reranker](https://huggingface.co/anonymousauthor01/instruction_following_reranker)**

- **Architecture:** Llama-Nemotron-Rerank-1B-v2 (sequence classification)
- **Parameters:** 1.24B
- **License:** Apache 2.0
- **Training:** On-policy GRPO distillation from a 4B teacher (Stage 2 of the paper)

## Datasets

Both evaluation datasets are hosted on HuggingFace and are downloaded automatically by `scripts/evaluation/run_eval.sh`.

| Dataset | HuggingFace repo | Local default path |
|---------|------------------|--------------------|
| Validation train/val set | [anonymousauthor01/emnlp-2026-ifr-train-val-set](https://huggingface.co/datasets/anonymousauthor01/emnlp-2026-ifr-train-val-set) | `datasets/train-val-set` |
| MAIR OOD subsets | [anonymousauthor01/emnlp-2026-ifr-mair-ood](https://huggingface.co/datasets/anonymousauthor01/emnlp-2026-ifr-mair-ood) | `datasets/MAIR_OOD` |

Manual download commands:

```bash
huggingface-cli download anonymousauthor01/emnlp-2026-ifr-train-val-set \
  --repo-type dataset \
  --local-dir datasets/train-val-set

huggingface-cli download anonymousauthor01/emnlp-2026-ifr-mair-ood \
  --repo-type dataset \
  --local-dir datasets/MAIR_OOD
```

## Hardware Requirements

### Evaluation (this repository)
- **Minimum:** Any GPU with ≥4 GB VRAM (e.g., V100, A100, H100)
- **VRAM footprint:** ~4 GB for the 1B model in bfloat16
- **Runtime:** ~20 minutes for the full evaluation suite (val + MAIR)

### Training (to be released upon acceptance)
| Component | Minimum Hardware | Approximate Time |
|-----------|------------------|------------------|
| **Stage 1:** Teacher off-policy GRPO | 2× H200 (1 for actor, 1 for LLM judge OSS-20B) | ~3 days 23 hours |
| **Stage 2:** Student on-policy distillation | 1× H100 (or 2× A100 80GB) | ~4 hours 50 minutes |
| **Full evaluation suite** | 1× V100/A100/H100 (≥4 GB VRAM) | ~20 minutes |
| **Total reported pipeline** | — | ~101 H200 GPU-hours |

## What Will Be Released Upon Acceptance

Upon acceptance, we will release the complete reproduction package including:

- **Full training scripts** for both Stage 1 (teacher GRPO) and Stage 2 (on-policy distillation)
- **Data preprocessing pipeline** for constructing the instruction-following reranking benchmark from the 8 source datasets
- **LLM-judge prompt templates** used for Stage 1 reward signal generation
- **All ablation checkpoints** (A1–A8) and evaluation scripts
- **Hyperparameter configuration files** for all reported experiments

## Validation Datasets

The combined validation benchmark spans 8 instruction-following reranking datasets:

| Dataset | Queries | Domain |
|---------|---------|--------|
| FollowIR (TREC) | 49 | Web search |
| InstructIR | 991 | Instruction-following retrieval |
| InfoSearch | 477 | Information seeking |
| InfIR (MS MARCO) | 3,876 | Web search |
| InfIR (MetaMath) | 710 | Math |
| InfIR (LeetCode) | 254 | Code |
| InfIR (Robust04) | 196 | News |
| M-BEIR (WebQA) | 3,308 | Multi-hop |
| **Total** | **9,861** | |

## MAIR OOD Subsets

The 11 MAIR subsets used for out-of-distribution evaluation span:

| Category | Subsets |
|----------|---------|
| Ad hoc / exploratory retrieval | Core_2017, DD_2016 |
| FAQ / duplicate-question matching | Quora |
| Scientific / biomedical evidence | SciFact, SciDocs, Trec-Covid, NFCorpus, LitSearch |
| Financial domain retrieval | FiQA |
| Argumentative / viewpoint retrieval | ArguAna, Touche |

## Expected Output

After running `bash scripts/evaluation/run_eval.sh`, you should see output similar to:

```text
  --- VALIDATION (9861 queries) ---
  nDCG@6: 0.7624  SD=0.3730  95% CI [0.7550, 0.7700]
  MRR@6:  0.7475  SD=0.3885  95% CI [0.7400, 0.7550]

  --- MAIR (OOD) (869 queries) ---
  nDCG@6: 0.7670  SD=0.3274  95% CI [0.7450, 0.7890]
  MRR@6:  0.8289  SD=0.3323  95% CI [0.8070, 0.8510]
```

Small numerical differences (<0.001) may arise from floating-point precision across hardware and library versions.

## Citation

*Citation will be provided upon acceptance.*

