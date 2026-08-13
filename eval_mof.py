#!/usr/bin/env python3
"""Проверка качества FST-Net 3B 1-bit MoF (JARVIS Core).

Два режима:
  1. loss на валидации (--val)  — из data/jarvis_full.json
  2. генерация на выборке промптов (--gen N) — печатает N ответов + метрики формата

Запуск (Colab / локально):
  !python eval_mof.py --ckpt checkpoints/3b_mof/best.pt --gen 10
  !python eval_mof.py --ckpt checkpoints/3b_mof/best.pt --val
Env: FSTNET_EVAL_N, FSTNET_GEN_N
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

def log(msg): print(msg, flush=True)

sys.path.insert(0, os.getcwd())

from config_3b_mof import FSTMoFConfig
from model.core_mof import FSTMoFModel
from tokenizers import Tokenizer

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
IGNORE = -100
PAD = 0


def encode(tok, text):
    return tok.encode(f"{IM_S}assistant\n{text}{IM_E}").ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("FSTNET_CKPT",
                    "checkpoints/3b_mof/best.pt"))
    ap.add_argument("--data", default="data/jarvis_full.json")
    ap.add_argument("--gen", type=int, default=int(os.environ.get("FSTNET_GEN_N", "0")))
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        log(f"[FAIL] чекпоинт не найден: {args.ckpt}")
        sys.exit(1)

    cfg = FSTMoFConfig()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_ck = ck.get("config", None)
    if cfg_ck is not None:
        for k, v in vars(cfg_ck).items():
            setattr(cfg, k, v)

    model = FSTMoFModel(cfg).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.set_binarize(1.0)
    model.eval()
    step = ck.get("step", "?")
    params = sum(p.numel() for p in model.parameters())
    log(f"Loaded {args.ckpt} | step={step} | params={params/1e9:.3f}B")

    tok = Tokenizer.from_file(cfg.tokenizer_path)

    if args.val:
        data = json.load(open(args.data))
        crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")
        total, n = 0.0, 0
        t0 = time.time()
        with torch.no_grad():
            for conv in data:
                roles = [(r, c) for r, c in conv]
                ids = []
                lm = []
                for role, content in roles:
                    seg = tok.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
                    if len(ids) + len(seg) > cfg.max_seq_len:
                        break
                    ids += seg
                    lm += [1 if role == "assistant" else 0] * len(seg)
                if len(ids) < 8:
                    continue
                x = torch.tensor([ids[:-1]], device=args.device)
                y = torch.tensor([ids[1:]], device=args.device)
                yt = torch.full_like(y, IGNORE)
                yt[0, 1:] = torch.where(torch.tensor(lm[1:], dtype=torch.bool), y[0, 1:], IGNORE)
                logits, _ = model(x)
                total += crit(logits.view(-1, cfg.vocab_size), yt.view(-1)).item()
                n += (yt != IGNORE).sum().item()
        val = total / max(n, 1)
        log(f"VAL CE: {val:.4f} | tokens={n} | {time.time()-t0:.0f}s")

    if args.gen > 0:
        samples = json.load(open(args.data))[:args.gen]
        n_sir = n_think = n_tool = 0
        t0 = time.time()
        total_new = 0
        for i, conv in enumerate(samples):
            user = next((c for r, c in conv if r == "user"), "")
            prompt = f"{IM_S}system\nSystem: JARVIS{IM_E}\n{IM_S}user\n{user}{IM_E}\n{IM_S}assistant\n"
            ids = tok.encode(prompt).ids
            if not ids:
                ids = [1]
            x = torch.tensor([ids], device=args.device)
            out = model.generate(x, max_new=args.max_new, temperature=0.7, top_k=40, top_p=0.9)
            new_ids = out[0, len(ids):].tolist()
            total_new += len(new_ids)
            text = tok.decode(new_ids)
            text = text.split(IM_E)[0].strip()
            s = text.lower()
            n_sir += ("сэр" in s or "sir" in s)
            n_think += ("<think>" in text)
            n_tool += ("<tool_call>" in text)
            log(f"--- [{i}] user: {user[:60]!r}")
            log(text[:400])
        el = time.time() - t0
        toks = max(total_new, 1)
        log(f"GEN METRICS: Sir={n_sir}/{args.gen} think={n_think}/{args.gen} "
            f"tool_call={n_tool}/{args.gen} | {toks/el:.1f} tok/s (raw fwd)")


if __name__ == "__main__":
    main()