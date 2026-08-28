# On-Policy Distillation Meets Off-Policy GRPO: Training Sparse Instruction-Following Rerankers that Beat Their Teachers

**Anonymous EMNLP 2026 Submission**

---

## Overview

This repository provides the evaluation code, the Stage 2 training script, dataset download hooks, and model checkpoint necessary to reproduce the main results reported in the paper. The final **Distilled-1B** reranker achieves:

| Benchmark | nDCG@6 | MRR@6 | 95% CI (nDCG) |
|-----------|--------|-------|---------------|
| Validation (9,861 queries) | 0.7624 | 0.7475 | [0.755, 0.770] |
| MAIR OOD (869 queries) | 0.7670 | 0.8289 | [0.755, 0.779] |

## Results

All numbers are nDCG@6 / MRR@6 with bootstrap 95% CIs (10,000 resamples).

### Main validation results (9,861 instruction-following queries)

**Distilled-1B achieves the best overall nDCG@6 among all evaluated models while remaining compact.** It exceeds Cohere Rerank v3.5, Cohere Rerank v4.0-fast, Rank-R1-7B, and REARANK-7B on validation nDCG@6; Jina Reranker v2 obtains a slightly higher MRR@6 point estimate.

| Model | Params | nDCG@6 | SD | MRR@6 | SD |
|-------|:------:|--------|:----:|--------|:----:|
| Qwen3-Reranker-4B | 4B | 0.6564 [0.649, 0.664] | 0.3821 | 0.6387 [0.631, 0.647] | 0.4055 |
| Base-1B (no training) | 1B | 0.6972 [0.690, 0.705] | 0.3657 | 0.6639 [0.656, 0.672] | 0.3909 |
| ZeRank-2 base (4B) | 4B | 0.7222 [0.715, 0.730] | 0.3516 | 0.7056 [0.698, 0.714] | 0.3704 |
| BGE Reranker v2 M3 | 568M | 0.7310 [0.724, 0.739] | 0.3697 | 0.7057 [0.698, 0.714] | 0.3897 |
| Teacher GRPO (4B) | 4B | 0.7422 [0.735, 0.750] | 0.3716 | 0.7256 [0.718, 0.734] | 0.3904 |
| Cohere Rerank v3.5 | — | 0.7459 [0.7387, 0.7533] | 0.3711 | 0.7265 [0.7190, 0.7343] | 0.3889 |
| REARANK-7B | 7B | 0.7486 [0.7411, 0.7538] | 0.3626 | 0.7338 [0.7242, 0.7376] | 0.3766 |
| Cohere Rerank v4.0-fast | — | 0.7494 [0.7422, 0.7569] | 0.3708 | 0.7325 [0.7251, 0.7404] | 0.3875 |
| Rank-R1-7B | 7B | 0.7506 [0.7432, 0.7552] | 0.3722 | 0.7351 [0.7266, 0.7394] | 0.3827 |
| Jina Reranker v2 | 278M | 0.7605 [0.753, 0.768] | 0.3737 | **0.7497 [0.742, 0.758]** | 0.3884 |
| **Distilled-1B (ours)** | **1B** | **0.7624 [0.755, 0.770]** | 0.3730 | 0.7475 [0.740, 0.755] | 0.3885 |

### Out-of-distribution generalization on MAIR-11 (869 queries)

The 11-subset, 869-query MAIR evaluation, including the training ablations (A1–A10). **A3: On-policy GRPO from Teacher** is the released Distilled-1B model.

| Model | nDCG@6 | SD | MRR@6 | SD |
|-------|--------|:----:|--------|:----:|
| Base-1B (no training) | 0.7119 [0.698, 0.726] | 0.2150 | 0.7771 [0.762, 0.792] | 0.2250 |
| Teacher GRPO (4B) | 0.6880 [0.674, 0.702] | 0.2150 | 0.7312 [0.716, 0.747] | 0.2350 |
| REARANK-7B | 0.7315 [0.717, 0.746] | 0.2180 | 0.7987 [0.783, 0.814] | 0.2300 |
| Rank-R1-7B | 0.7342 [0.720, 0.748] | 0.2120 | 0.8092 [0.794, 0.824] | 0.2220 |
| A1: Offline KD | 0.7212 [0.707, 0.735] | 0.2080 | 0.7860 [0.771, 0.801] | 0.2200 |
| A2: Off-policy GRPO | 0.7356 [0.722, 0.749] | 0.2050 | 0.8035 [0.789, 0.818] | 0.2150 |
| A8: On-policy GRPO from base ZeRank-2 | 0.7412 [0.728, 0.754] | 0.1980 | 0.8036 [0.790, 0.817] | 0.2050 |
| **A3: On-policy GRPO from Teacher (ours)** | **0.7670 [0.755, 0.779]** | 0.1850 | 0.8289 [0.816, 0.842] | 0.1900 |
| A4: No KL term | 0.7688 [0.757, 0.781] | 0.1820 | 0.8282 [0.816, 0.841] | 0.1880 |
| A5: No entropy | 0.7665 [0.754, 0.779] | 0.1860 | 0.8301 [0.817, 0.843] | 0.1900 |
| A6: Hard labels | 0.7365 [0.723, 0.750] | 0.2080 | 0.8000 [0.786, 0.814] | 0.2150 |
| A9: On-policy GKD | 0.7386 [0.726, 0.751] | 0.1920 | 0.8088 [0.796, 0.822] | 0.1950 |
| A10: RankNet pairwise KD | 0.7416 [0.729, 0.754] | 0.1900 | 0.8128 [0.800, 0.826] | 0.1920 |

Three patterns stand out. First, RL-based training substantially improves over the untrained 1B backbone, raising validation nDCG@6 from 0.6972 to 0.7624. Second, the distilled student exceeds both the untrained 4B teacher backbone and the Stage 1 teacher. Third, Distilled-1B remains competitive with strong external rerankers despite its compact size, and reduces mean latency from 27.0 ms (4B teacher) to 9.2 ms.

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

### 5. Train the Distilled-1B model (Stage 2)

Reproduce the on-policy GRPO distillation that produces the released Distilled-1B
reranker (row **A3** in the MAIR ablation table):

```bash
bash scripts/training/launch.sh
```

This downloads the train/val split from HuggingFace (if absent), loads the
frozen teacher and the trainable student, and trains for one epoch, periodically
evaluating student nDCG@6 and saving the top-k checkpoints. All hyperparameters
match the paper and are overridable via environment variables — e.g. to point at
your own Stage 1 teacher checkpoint:

```bash
TEACHER_PATH=/path/to/stage1_teacher_ckpt \
  bash scripts/training/launch.sh
```

See [`scripts/training/README.md`](scripts/training/README.md) for the full
argument reference and the training objective.

## Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies (pinned versions)
├── scripts/
│   ├── training/
│   │   ├── on_policy_rl.py                # Stage 2: on-policy GRPO distillation
│   │   ├── launch.sh                      # One-command training launcher
│   │   └── README.md                      # Training objective + argument reference
│   └── evaluation/
│       ├── evaluate.py                    # Evaluation script
│       └── run_eval.sh                    # One-command launcher
├── datasets/                              # Created automatically if absent
│   ├── train-val-set/                     # HF snapshot: emnlp-2026-ifr-train-val-set
│   │   ├── train.jsonl                    # Training queries (Stage 2)
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
└── results/                               # Output directory created by the launchers
    ├── eval.log                           # Evaluation console log
    ├── evaluation_results.json            # Full evaluation metrics
    └── distill_stage2/                    # Stage 2 training outputs
        ├── checkpoints/                   # Top-k student checkpoints
        ├── best_checkpoints.json
        ├── final_metrics.json
        └── train.log
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

### Training
The Stage 2 distillation script (`scripts/training/`) is included in this repository.

| Component | Minimum Hardware | Approximate Time |
|-----------|------------------|------------------|
| **Stage 1:** Teacher off-policy GRPO | 2× H200 (1 for actor, 1 for LLM judge OSS-20B) | ~3 days 23 hours |
| **Stage 2:** Student on-policy distillation | 1× H200 (or 2× A100 80GB) | ~4 hours 50 minutes |
| **Full evaluation suite** | 1× V100/A100/H100/H200 (≥4 GB VRAM) | ~20 minutes |
| **Total reported pipeline** | — | ~101 H200 GPU-hours |

## What Will Be Released Upon Acceptance

This repository already includes the **Stage 2 on-policy distillation training
script** (`scripts/training/`) and the **full evaluation suite**
(`scripts/evaluation/`). Upon acceptance, we will additionally release:

- **Stage 1 training code** (teacher off-policy GRPO)
- **Data preprocessing pipeline** for constructing the instruction-following reranking benchmark from the 8 source datasets
- **LLM-judge prompt templates** used for Stage 1 reward signal generation
- **All ablation checkpoints** (A1–A10) and their configuration files
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

## Note

Small numerical differences (<0.001) may arise from floating-point precision across hardware and library versions.

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{prabhakar2026onpolicy,
  title     = {On-Policy Distillation Meets Off-Policy {GRPO}:
               Training Compact Instruction-Following Rerankers},
  author    = {Prabhakar, Vignesh and Pan, Jialing and
               Ankisettipalli, Anil Babu},
  booktitle = {Findings of the Association for Computational Linguistics:
               EMNLP 2026},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```

