#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — Reproducible evaluation for the Distilled-1B reranker
=====================================================================
Evaluates the instruction-following reranker on:
  (1) Validation benchmark (9,861 queries across 8 datasets)
  (2) MAIR OOD benchmark (869 queries across 11 subsets)

By default, this script downloads both evaluation datasets from HuggingFace:
  - anonymousauthor01/emnlp-2026-ifr-train-val-set
  - anonymousauthor01/emnlp-2026-ifr-mair-ood

Reports nDCG@6, MRR@6 with bootstrap 95% CIs, per-dataset breakdown.

Requirements:
  pip install torch transformers datasets huggingface_hub numpy

Hardware:
  Any GPU with >=4GB VRAM (V100, A100, H100, etc.)
  Evaluation takes ~20 minutes on a single GPU.

Usage:
  python evaluate.py \
    --model_path anonymousauthor01/instruction_following_reranker \
    --data_dir datasets \
    --out_dir results \
    --bf16

Optional local/offline usage:
  python evaluate.py \
    --model_path /path/to/model \
    --val_jsonl datasets/train-val-set/val.jsonl \
    --mair_root datasets/MAIR_OOD \
    --out_dir results \
    --bf16
"""

import argparse
import json
import math
import os
import random
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import torch


# ======================================================================
# Compatibility
# ======================================================================
def patch_transformers_cache_utils():
    """Patch for older transformers versions missing HybridCache."""
    try:
        import transformers.cache_utils as cu
        if hasattr(cu, "Cache") and not hasattr(cu, "HybridCache"):
            cu.HybridCache = cu.Cache
    except Exception:
        pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# HuggingFace dataset download helpers
# ======================================================================
DEFAULT_VAL_DATASET_ID = "anonymousauthor01/emnlp-2026-ifr-train-val-set"
DEFAULT_MAIR_DATASET_ID = "anonymousauthor01/emnlp-2026-ifr-mair-ood"


def hf_token() -> Optional[str]:
    """Use either HF_TOKEN or HUGGINGFACE_HUB_TOKEN when available."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def snapshot_dataset(repo_id: str, local_dir: str, force_download: bool = False) -> str:
    """Download a HuggingFace dataset repo if it is not already present."""
    from huggingface_hub import snapshot_download

    local_path = Path(local_dir)
    marker_files = list(local_path.rglob("*")) if local_path.exists() else []
    if marker_files and not force_download:
        print(f"  Using cached dataset at {local_path}")
        return str(local_path)

    print(f"  Downloading dataset {repo_id} -> {local_path}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        token=hf_token(),
        force_download=force_download,
    )
    return str(local_path)


def find_val_jsonl(root: str) -> str:
    """Find val.jsonl under a downloaded train/val dataset repo."""
    root_path = Path(root)
    candidates = [
        root_path / "val.jsonl",
        root_path / "train-val-set" / "val.jsonl",
        root_path / "datasets" / "train-val-set" / "val.jsonl",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    matches = sorted(root_path.rglob("val.jsonl"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        f"Could not find val.jsonl under {root_path}. "
        "Check that the HF dataset repo was downloaded correctly."
    )


def resolve_mair_root(root: str) -> str:
    """
    Resolve the MAIR root containing docs/ and queries/.
    Supports either:
      root/docs/... and root/queries/...
    or:
      root/MAIR_OOD/docs/... and root/MAIR_OOD/queries/...
    """
    root_path = Path(root)
    candidates = [
        root_path,
        root_path / "MAIR_OOD",
        root_path / "datasets" / "MAIR_OOD",
    ]
    for c in candidates:
        if (c / "docs").is_dir() and (c / "queries").is_dir():
            return str(c)

    for c in root_path.rglob("*"):
        if c.is_dir() and (c / "docs").is_dir() and (c / "queries").is_dir():
            return str(c)

    raise FileNotFoundError(
        f"Could not find MAIR docs/ and queries/ directories under {root_path}. "
        "Check that the HF dataset repo was downloaded correctly."
    )


# ======================================================================
# Metrics
# ======================================================================
def _dcg(rels: List[float]) -> float:
    return sum(((2.0 ** r) - 1.0) / math.log2(2.0 + i) for i, r in enumerate(rels))


def ndcg_at_k(order: List[int], rels: List[float], k: int) -> float:
    if not rels:
        return 0.0
    k = min(k, len(rels))
    picked = order[:k]
    if not picked:
        return 0.0
    dcg = _dcg([float(rels[i]) for i in picked])
    ideal = sorted([float(x) for x in rels], reverse=True)[:k]
    idcg = _dcg(ideal)
    return float(dcg / idcg) if idcg > 1e-12 else 0.0


def mrr_at_k(order: List[int], rels: List[float], k: int, thr: float = 1e-9) -> float:
    k = min(k, len(order))
    for rank, idx in enumerate(order[:k], start=1):
        if float(rels[idx]) > thr:
            return 1.0 / float(rank)
    return 0.0


def bootstrap_ci(values: List[float], n_resamples: int = 10000, seed: int = 42):
    """Percentile bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    vals = np.array(values)
    n = len(vals)
    means = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        means[i] = vals[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ======================================================================
# Data loading
# ======================================================================
_LABEL_KEYS = ["label", "labels", "relevance", "rel", "score", "judgement", "grade"]


def extract_rels(row: Dict) -> Optional[List[float]]:
    cands = row.get("candidates")
    if not isinstance(cands, list) or len(cands) < 2:
        return None
    rels, found = [], False
    for c in cands:
        if not isinstance(c, dict):
            return None
        v = None
        for k in _LABEL_KEYS:
            if k in c:
                v = c[k]
                break
        if v is None:
            rels.append(0.0)
            continue
        found = True
        if isinstance(v, (int, float)):
            rels.append(float(v))
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            rels.append(float(v[0]))
        else:
            try:
                rels.append(float(v))
            except Exception:
                rels.append(0.0)
    return rels if found else None


def extract_texts(row: Dict) -> Optional[List[str]]:
    cands = row.get("candidates")
    if not isinstance(cands, list) or len(cands) < 2:
        return None
    texts = []
    for c in cands:
        if isinstance(c, dict):
            t = c.get("text", "")
            texts.append(str(t) if t else "")
        else:
            texts.append("")
    return texts if sum(1 for t in texts if t.strip()) >= 2 else None


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def make_query_str(instruction: str, query: str) -> str:
    instruction = (instruction or "").strip()
    query = (query or "").strip()
    if instruction:
        return f"{instruction}\n\n{query}".strip()
    return query.strip()


# ======================================================================
# MAIR data loader
# ======================================================================
MAIR_SUBSETS = [
    "ArguAna", "Core_2017", "DD_2016", "FiQA", "LitSearch",
    "NFCorpus", "Quora", "SciDocs", "SciFact", "Trec-Covid", "Touche"
]


def load_mair_data(mair_root: str) -> List[Dict]:
    """Load MAIR OOD evaluation subsets from a local snapshot."""
    from datasets import load_from_disk

    rows = []
    for sub in MAIR_SUBSETS:
        docs_path = os.path.join(mair_root, "docs", sub, "docs")
        queries_path = os.path.join(mair_root, "queries", sub, "queries")
        if not os.path.isdir(docs_path) or not os.path.isdir(queries_path):
            print(f"  [skip] MAIR {sub}: not found at {queries_path}")
            continue

        docs_ds = load_from_disk(docs_path)
        queries_ds = load_from_disk(queries_path)
        doc_index = {str(i): str(t) for i, t in zip(docs_ds["id"], docs_ds["doc"])}
        doc_ids = list(docs_ds["id"])

        rng = random.Random(42)
        count = 0
        for qi in range(len(queries_ds)):
            qrow = queries_ds[qi]
            query = str(qrow.get("query", "") or "")
            instruction = str(qrow.get("instruction", "") or "")
            labels = qrow.get("labels", []) or []
            if not query.strip():
                continue

            seen = set()
            cands = []
            for x in labels:
                did = str(x.get("id", "") or "")
                if not did or did in seen:
                    continue
                txt = doc_index.get(did)
                if not txt or not txt.strip():
                    continue
                seen.add(did)
                cands.append({"text": txt, "label": float(x.get("score", 0) or 0)})

            # Pad with negatives if fewer than 6 candidates.
            if len(cands) < 6:
                pool = [d for d in doc_ids if str(d) not in seen]
                rng.shuffle(pool)
                for did in pool[:max(0, 6 - len(cands))]:
                    txt = doc_index.get(str(did))
                    if txt and txt.strip():
                        cands.append({"text": txt, "label": 0.0})

            if len(cands) >= 2:
                rows.append({
                    "query": query,
                    "instruction": instruction,
                    "dataset": f"MAIR_{sub}",
                    "candidates": cands,
                })
                count += 1

        print(f"  MAIR {sub}: {count} queries")
    return rows


# ======================================================================
# Model scorer
# ======================================================================
class RerankerScorer:
    """
    Loads and scores with the Distilled-1B instruction-following reranker.

    Architecture: Llama-Nemotron-Rerank-1B-v2 (sequence classification).
    Handles both custom auto_map models and standard HF seq classifiers.
    """

    def __init__(self, model_path: str, device: torch.device, bf16: bool = True):
        from transformers import AutoTokenizer, AutoConfig
        patch_transformers_cache_utils()

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True, token=hf_token())
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, token=hf_token())
        has_auto_map = (
            hasattr(cfg, "auto_map")
            and isinstance(cfg.auto_map, dict)
            and "AutoModelForSequenceClassification" in cfg.auto_map
        )

        if has_auto_map:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            ref = cfg.auto_map["AutoModelForSequenceClassification"]
            cls = get_class_from_dynamic_module(ref, model_path, token=hf_token())
            self.model = cls.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=(torch.bfloat16 if bf16 else None),
                token=hf_token(),
            ).to(device).eval()
            print(f"  Loaded via auto_map: {type(self.model).__name__}")
        else:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=(torch.bfloat16 if bf16 else None),
                token=hf_token(),
            ).to(device).eval()
            print("  Loaded via AutoModelForSequenceClassification")

        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  VRAM used: {vram:.1f} GB")

    @torch.no_grad()
    def score(self, instruction: str, query: str, cand_texts: List[str],
              max_len: int = 512, chunk: int = 16) -> List[float]:
        q = make_query_str(instruction, query)
        docs = [str(x or "") for x in cand_texts]
        outs = []
        for i in range(0, len(docs), chunk):
            sub = docs[i:i + chunk]
            enc = self.tok(
                [q] * len(sub), sub,
                padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            if logits.dim() == 2 and logits.size(-1) == 1:
                s = logits[:, 0]
            elif logits.dim() == 2 and logits.size(-1) >= 2:
                s = logits[:, -1]
            else:
                s = logits.view(logits.size(0), -1)[:, 0]
            outs.extend(s.float().tolist())
        return outs


# ======================================================================
# Main evaluation
# ======================================================================
def evaluate(rows: List[Dict], scorer: RerankerScorer, k: int = 6, max_len: int = 512):
    """Score all queries, return per-query metrics and dataset labels."""
    results = []
    skipped = 0
    t0 = time.time()

    for idx, row in enumerate(rows):
        texts = extract_texts(row)
        rels = extract_rels(row)
        if texts is None or rels is None:
            skipped += 1
            continue

        instruction = str(row.get("instruction", "") or "")
        query = str(row.get("query", "") or "")
        scores = scorer.score(instruction, query, texts, max_len=max_len)
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)

        results.append({
            "dataset": str(row.get("dataset", "")),
            "ndcg": ndcg_at_k(order, rels, k),
            "mrr": mrr_at_k(order, rels, k),
        })

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Scored {idx + 1}/{len(rows)} "
                  f"({elapsed:.0f}s, {(idx + 1) / max(elapsed, 1e-9):.0f} q/s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} scored, {skipped} skipped, {elapsed:.0f}s")
    return results


def print_results(results: List[Dict], k: int, label: str):
    """Print aggregate metrics with bootstrap CIs."""
    if not results:
        print(f"\n  {label}: no results")
        return {}

    ndcgs = [r["ndcg"] for r in results]
    mrrs = [r["mrr"] for r in results]
    ndcg_mean = np.mean(ndcgs)
    mrr_mean = np.mean(mrrs)
    ndcg_sd = np.std(ndcgs, ddof=1)
    mrr_sd = np.std(mrrs, ddof=1)
    ndcg_lo, ndcg_hi = bootstrap_ci(ndcgs)
    mrr_lo, mrr_hi = bootstrap_ci(mrrs)

    print(f"\n  --- {label} ({len(results)} queries) ---")
    print(f"  nDCG@{k}: {ndcg_mean:.4f}  SD={ndcg_sd:.4f}  "
          f"95% CI [{ndcg_lo:.4f}, {ndcg_hi:.4f}]")
    print(f"  MRR@{k}:  {mrr_mean:.4f}  SD={mrr_sd:.4f}  "
          f"95% CI [{mrr_lo:.4f}, {mrr_hi:.4f}]")

    return {
        "n": len(results),
        f"ndcg@{k}": round(ndcg_mean, 4),
        f"ndcg@{k}_sd": round(ndcg_sd, 4),
        f"ndcg@{k}_ci": [round(ndcg_lo, 4), round(ndcg_hi, 4)],
        f"mrr@{k}": round(mrr_mean, 4),
        f"mrr@{k}_sd": round(mrr_sd, 4),
        f"mrr@{k}_ci": [round(mrr_lo, 4), round(mrr_hi, 4)],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate the Distilled-1B instruction-following reranker "
                    "on the validation benchmark and MAIR OOD subsets.")
    ap.add_argument("--model_path", required=True,
                    help="HuggingFace model ID or local path "
                         "(e.g. anonymousauthor01/instruction_following_reranker)")

    # New default HF-backed workflow.
    ap.add_argument("--data_dir", default="./datasets",
                    help="Directory where HF dataset snapshots are cached/downloaded")
    ap.add_argument("--val_dataset_id", default=DEFAULT_VAL_DATASET_ID,
                    help="HF dataset repo containing val.jsonl")
    ap.add_argument("--mair_dataset_id", default=DEFAULT_MAIR_DATASET_ID,
                    help="HF dataset repo containing MAIR_OOD docs/ and queries/")
    ap.add_argument("--force_download", action="store_true",
                    help="Force re-download of HF dataset snapshots")

    # Backward-compatible local/offline overrides.
    ap.add_argument("--val_jsonl", default="",
                    help="Optional local path to val.jsonl. If omitted, downloaded from HF.")
    ap.add_argument("--mair_root", default="",
                    help="Optional local path to MAIR_OOD directory. If omitted, downloaded from HF.")
    ap.add_argument("--skip_mair", action="store_true",
                    help="Only evaluate the validation set")

    ap.add_argument("--out_dir", default="./results",
                    help="Directory for output JSON")
    ap.add_argument("--k", type=int, default=6,
                    help="Cutoff for nDCG@k and MRR@k (default: 6)")
    ap.add_argument("--max_len", type=int, default=512,
                    help="Max sequence length for tokenization")
    ap.add_argument("--bf16", action="store_true",
                    help="Use bfloat16 precision (recommended)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="",
                    help="Device (default: auto-detect cuda/cpu)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Resolve/download data ----
    data_dir = Path(args.data_dir)
    if args.val_jsonl:
        val_jsonl = args.val_jsonl
    else:
        val_snapshot_dir = data_dir / "train-val-set"
        snapshot_dataset(args.val_dataset_id, str(val_snapshot_dir), args.force_download)
        val_jsonl = find_val_jsonl(str(val_snapshot_dir))

    mair_root = ""
    if not args.skip_mair:
        if args.mair_root:
            mair_root = resolve_mair_root(args.mair_root)
        else:
            mair_snapshot_dir = data_dir / "MAIR_OOD"
            snapshot_dataset(args.mair_dataset_id, str(mair_snapshot_dir), args.force_download)
            mair_root = resolve_mair_root(str(mair_snapshot_dir))

    # ---- Load data ----
    print(f"\nLoading validation set: {val_jsonl}")
    val_rows = read_jsonl(val_jsonl)
    print(f"  {len(val_rows)} queries")

    mair_rows = []
    if mair_root:
        print(f"\nLoading MAIR OOD: {mair_root}")
        mair_rows = load_mair_data(mair_root)
        print(f"  {len(mair_rows)} total MAIR queries")

    # ---- Load model ----
    print(f"\nLoading model: {args.model_path}")
    scorer = RerankerScorer(args.model_path, device, args.bf16)

    # ---- Evaluate validation ----
    print(f"\n{'=' * 60}")
    print(f"  EVALUATING: {args.model_path}")
    print(f"{'=' * 60}")

    print("\nScoring validation set...")
    val_results = evaluate(val_rows, scorer, args.k, args.max_len)
    val_summary = print_results(val_results, args.k, "VALIDATION")

    by_dataset = defaultdict(list)
    for r in val_results:
        by_dataset[r["dataset"]].append(r)

    print(f"\n  --- PER-DATASET VALIDATION BREAKDOWN ---")
    print(f"  {'Dataset':<25} {'n':>6} {f'nDCG@{args.k}':>10} {f'MRR@{args.k}':>10}")
    ds_summary = {}
    for ds in sorted(by_dataset.keys()):
        ds_res = by_dataset[ds]
        nd = np.mean([r["ndcg"] for r in ds_res])
        mr = np.mean([r["mrr"] for r in ds_res])
        print(f"  {ds:<25} {len(ds_res):>6} {nd:>10.4f} {mr:>10.4f}")
        ds_summary[ds] = {
            "n": len(ds_res),
            f"ndcg@{args.k}": round(nd, 4),
            f"mrr@{args.k}": round(mr, 4),
        }

    # ---- Evaluate MAIR ----
    mair_summary = {}
    mair_ds_summary = {}
    if mair_rows:
        print("\nScoring MAIR OOD set...")
        mair_results = evaluate(mair_rows, scorer, args.k, args.max_len)
        mair_summary = print_results(mair_results, args.k, "MAIR (OOD)")

        mair_by_ds = defaultdict(list)
        for r in mair_results:
            mair_by_ds[r["dataset"]].append(r)

        print(f"\n  --- PER-SUBSET MAIR BREAKDOWN ---")
        print(f"  {'Subset':<25} {'n':>6} {f'nDCG@{args.k}':>10} {f'MRR@{args.k}':>10}")
        for ds in sorted(mair_by_ds.keys()):
            ds_res = mair_by_ds[ds]
            nd = np.mean([r["ndcg"] for r in ds_res])
            mr = np.mean([r["mrr"] for r in ds_res])
            print(f"  {ds:<25} {len(ds_res):>6} {nd:>10.4f} {mr:>10.4f}")
            mair_ds_summary[ds] = {
                "n": len(ds_res),
                f"ndcg@{args.k}": round(nd, 4),
                f"mrr@{args.k}": round(mr, 4),
            }

    # ---- Save results ----
    output = {
        "model_path": args.model_path,
        "k": args.k,
        "seed": args.seed,
        "data": {
            "val_dataset_id": args.val_dataset_id if not args.val_jsonl else None,
            "val_jsonl": val_jsonl,
            "mair_dataset_id": args.mair_dataset_id if mair_root and not args.mair_root else None,
            "mair_root": mair_root or None,
        },
        "validation": val_summary,
        "validation_per_dataset": ds_summary,
    }
    if mair_summary:
        output["mair_ood"] = mair_summary
        output["mair_per_subset"] = mair_ds_summary

    out_path = os.path.join(args.out_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ---- Print expected results for verification ----
    print(f"\n{'=' * 60}")
    print("  EXPECTED RESULTS (from the paper)")
    print(f"{'=' * 60}")
    print("  Validation nDCG@6: 0.7624 [0.755, 0.770]")
    print("  Validation MRR@6:  0.7475 [0.740, 0.755]")
    if mair_summary:
        print("  MAIR nDCG@6:       0.7670 [0.745, 0.789]")
        print("  MAIR MRR@6:        0.8289 [0.807, 0.851]")
    print("\n  Small numerical differences (<0.001) may arise from")
    print("  floating-point precision across hardware/library versions.")


if __name__ == "__main__":
    main()

