#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: On-policy GRPO distillation of an instruction-following reranker.

A frozen teacher (a ZeRank-2-style CausalLM reranker scored via the "Yes"-token
logit) defines the reference relevances and the soft distillation target. A
trainable student (Llama-Nemotron-Rerank-1B, a sequence-classification
cross-encoder) is optimized with:

  loss = pg_loss + lambda_kl_distill * KL(student || teacher) - ent_coef * H(student)

where pg_loss is a GRPO-style policy gradient computed from `group_size`
Plackett-Luce sampled slates, each rewarded by nDCG@k against the teacher-
derived relevances, with a group-normalized advantage.

Both teacher and student score every (query, document) candidate independently;
neither the model IDs nor any dataset path is hard-coded — everything is passed
via CLI. See scripts/training/launch.sh for the configuration used in the paper.
"""

import argparse
import json
import math
import os
import random
import time
import shutil
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# ---------------------------
# Optional W&B
# ---------------------------
try:
    import wandb
    _WANDB_OK = True
except Exception:
    wandb = None
    _WANDB_OK = False


# ---------------------------
# Transformers compatibility patch for Nemotron code
# ---------------------------
def patch_transformers_cache_utils():
    try:
        import transformers.cache_utils as cu
        if hasattr(cu, "Cache") and not hasattr(cu, "HybridCache"):
            cu.HybridCache = cu.Cache
    except Exception:
        pass


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now() -> float:
    return time.time()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_save_dir(src_tmp: str, dst: str):
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.replace(src_tmp, dst)


# ---------------------------
# Ranking / reward helpers
# ---------------------------
def _dcg(rels: List[float]) -> float:
    dcg = 0.0
    for i, r in enumerate(rels):
        gain = (2.0 ** float(r)) - 1.0
        dcg += gain / math.log2(2.0 + i)
    return dcg


def ndcg_at_k_for_slate(chosen: List[int], rels_all: List[float], k: int) -> float:
    if not rels_all:
        return 0.0
    k = min(k, len(rels_all))
    picked = chosen[:k]
    if not picked:
        return 0.0
    picked_rels = [float(rels_all[i]) for i in picked]
    dcg = _dcg(picked_rels)
    ideal = sorted([float(x) for x in rels_all], reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum()


def safe_mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def safe_std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = safe_mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(max(v, 0.0))


def normalize_scores(scores: List[float], method: str) -> List[float]:
    if not scores:
        return scores
    method = (method or "rank").lower()
    s = [float(x) for x in scores]

    if method == "none":
        return [float(1.0 / (1.0 + math.exp(-x))) for x in s]

    if method == "rank":
        idx = sorted(range(len(s)), key=lambda i: s[i])
        ranks = [0.0] * len(s)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and s[idx[j + 1]] == s[idx[i]]:
                j += 1
            avg = (i + j) / 2.0
            for t in range(i, j + 1):
                ranks[idx[t]] = avg
            i = j + 1
        denom = max(1.0, float(len(s) - 1))
        return [r / denom for r in ranks]

    if method == "zsigmoid":
        m = safe_mean(s)
        sd = safe_std(s) + 1e-6
        out = []
        for x in s:
            z = (x - m) / sd
            out.append(float(1.0 / (1.0 + math.exp(-z))))
        return out

    return normalize_scores(scores, "rank")


def plackett_luce_sample_and_logprob(
    logits: torch.Tensor,
    k: int,
    gen: torch.Generator,
    temperature: float = 1.0,
) -> Tuple[List[int], torch.Tensor]:
    device = logits.device
    n = logits.shape[0]
    k = min(k, n)
    temp = max(1e-6, float(temperature))

    remaining = torch.arange(n, device=device)
    cur_logits = (logits / temp).clone()
    logp_total = torch.zeros((), device=device, dtype=torch.float32)

    chosen: List[int] = []
    for _ in range(k):
        lprobs = F.log_softmax(cur_logits, dim=-1)
        idx_in_remaining = torch.multinomial(lprobs.exp(), 1, generator=gen).item()
        chosen_idx = remaining[idx_in_remaining].item()
        chosen.append(chosen_idx)
        logp_total = logp_total + lprobs[idx_in_remaining]

        mask = torch.ones_like(cur_logits, dtype=torch.bool)
        mask[idx_in_remaining] = False
        cur_logits = cur_logits[mask]
        remaining = remaining[mask]
        if cur_logits.numel() == 0:
            break

    return chosen, logp_total


# ---------------------------
# Teacher (Zerank-2) scoring (CausalLM "Yes" logit)
# ---------------------------
def _format_zerank_messages(instruction: str, query: str, doc: str) -> List[Dict[str, str]]:
    sys = (f"{instruction}\n\n{query}").strip() if (instruction or "").strip() else (query or "").strip()
    usr = (doc or "").strip()
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def build_zerank_inputs(tok, instruction: str, query: str, cand_texts: List[str], max_len: int):
    texts = []
    for doc in cand_texts:
        msgs = _format_zerank_messages(instruction, query, doc)
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(t)
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc


def scores_from_causallm_outputs(
    logits_vocab: torch.Tensor,
    attention_mask: torch.Tensor,
    yes_token_id: int,
    scale_div: float = 5.0,
) -> torch.Tensor:
    last_pos = attention_mask.sum(dim=1) - 1
    bsz = logits_vocab.shape[0]
    batch_idx = torch.arange(bsz, device=logits_vocab.device)
    last_logits = logits_vocab[batch_idx, last_pos]  # [B, V]
    yes_logits = last_logits[:, yes_token_id]
    return yes_logits.to(torch.float32) / float(scale_div)


class TeacherZerank(torch.nn.Module):
    def __init__(self, model_path: str, bf16: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.bf16 = bf16

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True, padding_side="right")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.yes_token_id = self.tok.encode("Yes", add_special_tokens=False)[0]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if bf16 else None),
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def score(self, instruction: str, query: str, cand_texts: List[str], max_len: int, score_chunk_size: int) -> torch.Tensor:
        out_scores = []
        chunk = max(1, int(score_chunk_size))
        for i in range(0, len(cand_texts), chunk):
            sub = cand_texts[i:i+chunk]
            enc = build_zerank_inputs(self.tok, instruction, query, sub, max_len=max_len)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, use_cache=False)
            s = scores_from_causallm_outputs(out.logits, enc["attention_mask"], self.yes_token_id, scale_div=5.0)
            out_scores.append(s)
        return torch.cat(out_scores, dim=0)


# ---------------------------
# Student scorer (Llama-Nemotron-Rerank-1B) — loads the sequence-classification
# head via the model's auto_map (custom LlamaBidirectional code on the Hub).
# ---------------------------
class StudentScorer(torch.nn.Module):
    def __init__(self, model_id: str, bf16: bool, device: torch.device):
        super().__init__()
        patch_transformers_cache_utils()

        self.model_id = model_id
        self.device = device
        self.bf16 = bf16
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        torch_dtype = torch.bfloat16 if bf16 else None

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        auto_map = getattr(cfg, "auto_map", None) or {}
        print(f"[student] config_class={cfg.__class__.__name__} model_type={getattr(cfg,'model_type','')}", flush=True)
        print(f"[student] auto_map keys={list(auto_map.keys())}", flush=True)

        if "AutoModelForSequenceClassification" not in auto_map:
            raise RuntimeError(f"Expected auto_map['AutoModelForSequenceClassification'] but got keys={list(auto_map.keys())}")

        ref = auto_map["AutoModelForSequenceClassification"]
        if isinstance(ref, (list, tuple)):
            ref = ref[0]

        # ref MUST be like: "llama_bidirectional_model.LlamaBidirectionalForSequenceClassification"
        if not isinstance(ref, str) or "." not in ref:
            raise RuntimeError(f"Unexpected auto_map reference: {ref}")

        # Recent transformers expects class_reference=ref (not class_name/module_file)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module(
            class_reference=ref,
            pretrained_model_name_or_path=model_id,
            cache_dir=None,
            force_download=False,
            revision=None,
            local_files_only=False,
        )

        self.model = cls.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)

        print("[student] loaded via auto_map[AutoModelForSequenceClassification]", flush=True)
        self.model.train()

    def _scores_seq_style(self, q_list: List[str], d_list: List[str], max_len: int) -> torch.Tensor:
        enc = self.tok(
            q_list,
            d_list,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("Student output has no .logits; cannot score.")

        if logits.dim() == 2 and logits.size(-1) == 1:
            s = logits[:, 0]
        elif logits.dim() == 2 and logits.size(-1) >= 2:
            s = logits[:, -1]
        else:
            s = logits.view(logits.size(0), -1)[:, 0]

        return s.to(torch.float32)

    def score_candidates(
        self,
        instruction: str,
        query: str,
        cand_texts: List[str],
        max_len: int,
        score_chunk_size: int,
    ) -> torch.Tensor:
        q = (f"{instruction}\n\n{query}").strip() if (instruction or "").strip() else (query or "").strip()

        out_scores = []
        chunk = max(1, int(score_chunk_size))
        for i in range(0, len(cand_texts), chunk):
            q_list = [q] * len(cand_texts[i:i+chunk])
            d_list = [str(x or "") for x in cand_texts[i:i+chunk]]
            out_scores.append(self._scores_seq_style(q_list, d_list, max_len=max_len))
        return torch.cat(out_scores, dim=0)


# ---------------------------
# Checkpointing
# ---------------------------
def atomic_save_pretrained(model_to_save, tok, save_dir: str):
    tmp_dir = save_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    model_to_save.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="10GB")
    tok.save_pretrained(tmp_dir)
    atomic_save_dir(tmp_dir, save_dir)


# ---------------------------
# Eval (student ranks; teacher defines rels)
# ---------------------------
@torch.no_grad()
def evaluate(
    student: StudentScorer,
    teacher: TeacherZerank,
    rows: List[Dict[str, Any]],
    device: torch.device,
    max_len: int,
    k: int,
    score_norm: str,
    score_chunk_size: int,
    max_rows: int = 0,
) -> Dict[str, float]:
    ndcgs = []
    n = 0

    for row in rows:
        cands = row.get("candidates", [])
        if not isinstance(cands, list) or len(cands) < 2:
            continue

        instruction = str(row.get("instruction", "") or "")
        query = str(row.get("query", "") or "")
        cand_texts = [str(c.get("text", "") or "") for c in cands]
        if sum(1 for t in cand_texts if t.strip()) < 2:
            continue

        t_scores = teacher.score(instruction, query, cand_texts, max_len=max_len, score_chunk_size=score_chunk_size).tolist()
        rels = normalize_scores(t_scores, score_norm)

        s_scores = student.score_candidates(instruction, query, cand_texts, max_len=max_len, score_chunk_size=score_chunk_size)
        order = torch.argsort(s_scores, descending=True).tolist()
        ndcg = ndcg_at_k_for_slate(order[:k], rels, k)
        ndcgs.append(ndcg)

        n += 1
        if max_rows > 0 and n >= max_rows:
            break

    return {
        f"val/student_ndcg@{k}_wrt_teacher": float(safe_mean(ndcgs)),
        "val/rows_scored": float(len(ndcgs)),
    }


# ---------------------------
# Main training loop
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)

    ap.add_argument("--teacher_path", required=True)
    ap.add_argument("--student_id", default="nvidia/llama-nemotron-rerank-1b-v2")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--batch_size_rows", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=2)

    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--pl_temperature", type=float, default=1.0)

    ap.add_argument("--teacher_temp", type=float, default=1.0)
    ap.add_argument("--student_temp", type=float, default=1.0)
    ap.add_argument("--score_norm", type=str, default="rank", choices=["none", "rank", "zsigmoid"])

    ap.add_argument("--lambda_kl_distill", type=float, default=1.0)
    ap.add_argument("--ent_coef", type=float, default=0.01)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=4400)
    ap.add_argument("--save_every", type=int, default=4400)
    ap.add_argument("--save_top_k", type=int, default=3)

    ap.add_argument("--score_chunk_size", type=int, default=32)

    # W&B
    ap.add_argument("--wandb_project", type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default="")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)

    # W&B
    use_wandb = False
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_project:
        if not _WANDB_OK:
            raise RuntimeError("wandb requested but not installed. pip install wandb")
        use_wandb = True
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            entity=args.wandb_entity or None,
            config=vars(args),
        )

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    print(f"[data] train={len(train_rows)}/{len(train_rows)} val={len(val_rows)}/{len(val_rows)}", flush=True)
    print(f"[teacher] {args.teacher_path}", flush=True)
    print(f"[student] {args.student_id}", flush=True)

    teacher = TeacherZerank(args.teacher_path, bf16=args.bf16, device=device)
    student = StudentScorer(args.student_id, bf16=args.bf16, device=device)

    opt = torch.optim.AdamW(student.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_updates = math.ceil((len(train_rows) / max(1, args.batch_size_rows)) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = max(1, int(total_updates * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_updates - step) / float(max(1, total_updates - warmup_steps)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 777)

    best: List[Tuple[float, str]] = []

    def maybe_save(metric: float, step: int):
        nonlocal best
        if args.save_top_k <= 0:
            return
        ckpt_name = f"ckpt_step{step:07d}_valndcg{metric:.4f}"
        ckpt_dir = os.path.join(args.out_dir, "checkpoints", ckpt_name)

        should = False
        if len(best) < args.save_top_k:
            should = True
        else:
            worst_metric, _ = sorted(best, key=lambda x: x[0], reverse=True)[-1]
            should = metric > worst_metric

        if not should:
            return

        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        atomic_save_pretrained(student.model, student.tok, ckpt_dir)

        best.append((metric, ckpt_dir))
        best = sorted(best, key=lambda x: x[0], reverse=True)
        while len(best) > args.save_top_k:
            _, path = best.pop(-1)
            shutil.rmtree(path, ignore_errors=True)

        save_json(
            os.path.join(args.out_dir, "best_checkpoints.json"),
            [{"metric": float(m), "path": p} for (m, p) in best],
        )
        print(f"[save] saved: {ckpt_dir}", flush=True)

    global_step = 0
    micro_step = 0
    opt.zero_grad(set_to_none=True)

    last_t = now()
    last_micro = 0

    for ep in range(args.epochs):
        random.shuffle(train_rows)

        for start in range(0, len(train_rows), args.batch_size_rows):
            batch = train_rows[start:start + args.batch_size_rows]

            batch_loss = torch.zeros((), device=device, dtype=torch.float32)
            batch_pg = torch.zeros((), device=device, dtype=torch.float32)
            batch_kl = torch.zeros((), device=device, dtype=torch.float32)
            batch_ent = torch.zeros((), device=device, dtype=torch.float32)
            batch_r_mean = 0.0
            batch_r_std = 0.0
            n_rows_used = 0

            for row in batch:
                cands = row.get("candidates", [])
                if not isinstance(cands, list) or len(cands) < 2:
                    continue

                instruction = str(row.get("instruction", "") or "")
                query = str(row.get("query", "") or "")
                cand_texts = [str(c.get("text", "") or "") for c in cands]
                if sum(1 for t in cand_texts if t.strip()) < 2:
                    continue

                with torch.no_grad():
                    t_scores = teacher.score(
                        instruction, query, cand_texts,
                        max_len=args.max_len,
                        score_chunk_size=args.score_chunk_size
                    ).detach().to(torch.float32)

                rels = normalize_scores(t_scores.tolist(), args.score_norm)
                t_dist = F.softmax(t_scores / max(1e-6, args.teacher_temp), dim=-1).clamp_min(1e-12)

                s_scores = student.score_candidates(
                    instruction, query, cand_texts,
                    max_len=args.max_len,
                    score_chunk_size=args.score_chunk_size
                ).to(torch.float32)

                s_logp = F.log_softmax(s_scores / max(1e-6, args.student_temp), dim=-1)
                s_p = torch.exp(s_logp)
                kl_distill = torch.sum(s_p * (s_logp - torch.log(t_dist)))

                ent = entropy_from_logits(s_scores)

                chosen_logps: List[torch.Tensor] = []
                rewards: List[float] = []

                for _ in range(args.group_size):
                    chosen, logp = plackett_luce_sample_and_logprob(
                        s_scores, k=args.k, gen=gen, temperature=args.pl_temperature
                    )
                    r = ndcg_at_k_for_slate(chosen, rels, args.k)
                    chosen_logps.append(logp)
                    rewards.append(r)

                r_t = torch.tensor(rewards, device=device, dtype=torch.float32)
                r_mean = r_t.mean()
                r_std = r_t.std(unbiased=False)
                adv = (r_t - r_mean) / (r_std + 1e-6)

                logp_stack = torch.stack(chosen_logps).to(torch.float32)
                pg_loss = -(adv * logp_stack).mean()

                loss = pg_loss + args.lambda_kl_distill * kl_distill - args.ent_coef * ent

                batch_loss = batch_loss + loss
                batch_pg = batch_pg + pg_loss
                batch_kl = batch_kl + kl_distill
                batch_ent = batch_ent + ent
                batch_r_mean += float(r_mean.item())
                batch_r_std += float(r_std.item())
                n_rows_used += 1

            if n_rows_used == 0:
                continue

            batch_loss = batch_loss / n_rows_used
            batch_pg = batch_pg / n_rows_used
            batch_kl = batch_kl / n_rows_used
            batch_ent = batch_ent / n_rows_used
            batch_r_mean /= n_rows_used
            batch_r_std /= n_rows_used

            (batch_loss / max(1, args.grad_accum)).backward()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(student.model.parameters(), args.clip_grad)

                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    t = now()
                    dt = max(1e-9, t - last_t)
                    dm = micro_step - last_micro
                    sps = dm / dt
                    last_t = t
                    last_micro = micro_step

                    lr = sched.get_last_lr()[0]
                    msg = (
                        f"ep={ep} step={global_step}/{total_updates} "
                        f"loss={batch_loss.item():.4f} pg={batch_pg.item():.4f} "
                        f"kl={batch_kl.item():.4f} ent={batch_ent.item():.4f} "
                        f"r_mean={batch_r_mean:.3f} r_std={batch_r_std:.3f} "
                        f"lr={lr:.2e} micro_sps={sps:.2f} rows={n_rows_used}"
                    )
                    print(msg, flush=True)

                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": batch_loss.item(),
                                "train/pg_loss": batch_pg.item(),
                                "train/kl_distill": batch_kl.item(),
                                "train/entropy": batch_ent.item(),
                                "reward/mean": batch_r_mean,
                                "reward/std": batch_r_std,
                                "train/lr": lr,
                                "train/micro_sps": sps,
                                "epoch": ep,
                                "global_step": global_step,
                                "train/rows_in_micro": n_rows_used,
                            },
                            step=global_step,
                        )

                if args.eval_every > 0 and (global_step % args.eval_every == 0):
                    student.model.eval()
                    metrics = evaluate(
                        student=student,
                        teacher=teacher,
                        rows=val_rows,
                        device=device,
                        max_len=args.max_len,
                        k=args.k,
                        score_norm=args.score_norm,
                        score_chunk_size=args.score_chunk_size,
                    )
                    student.model.train()
                    print(f"[eval] {metrics}", flush=True)
                    if use_wandb:
                        wandb.log(metrics, step=global_step)

                if args.save_every > 0 and (global_step % args.save_every == 0):
                    student.model.eval()
                    metrics = evaluate(
                        student=student,
                        teacher=teacher,
                        rows=val_rows,
                        device=device,
                        max_len=args.max_len,
                        k=args.k,
                        score_norm=args.score_norm,
                        score_chunk_size=args.score_chunk_size,
                    )
                    student.model.train()
                    val_metric = float(metrics[f"val/student_ndcg@{args.k}_wrt_teacher"])
                    maybe_save(val_metric, global_step)
                    save_json(os.path.join(args.out_dir, "last_eval.json"), metrics)

                if global_step >= total_updates:
                    break

        if global_step >= total_updates:
            break

    student.model.eval()
    final_metrics = evaluate(
        student=student,
        teacher=teacher,
        rows=val_rows,
        device=device,
        max_len=args.max_len,
        k=args.k,
        score_norm=args.score_norm,
        score_chunk_size=args.score_chunk_size,
    )
    save_json(os.path.join(args.out_dir, "final_metrics.json"), final_metrics)
    print(f"[done] final_metrics={final_metrics}", flush=True)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

