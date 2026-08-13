#!/usr/bin/env python3
"""Обучение FST-Net 3B 1-bit MoF (JARVIS Core) в Colab.

Архитектура: model/core_mof.py (BitLinear+STE, ContinuousField, GQA+RoPE, гиперсеть).
Спецификация: SPEC_3B_MOF.md.

Фазы (по прогрессу шагов):
  S0 [p<0.15]  Warmup-dense: полный float, binarize_ratio=0
  S1 [0.15-0.6] STE-binarize: ratio линейно 0->1, A8 QDQ с ratio>=0.5
  S2 [>0.6]    Orth-fine: W0 заморожен, поля+гиперсеть с L_orth

Запуск в Colab:
  from google.colab import drive; drive.mount('/content/drive')
  %cd fstnet
  !pip install -q tokenizers tqdm
  # 1) датасет: дистилляция с учителя (Qwen 27B 1-bit, llama.cpp server :8001)
  !python distill_colab.py --count 200000 --workers 8
  !python distill_colab.py --to-json --synthetic 40000
  # 2) обучение Stage 1 (общая база) -> чекпоинт checkpoints/3b_mof/moF_best.pt
  !FSTNET_EPOCHS=4 FSTNET_LR=2e-4 python train_colab_mof.py
  # 3) обучение Stage 2 (спец. датасет, W0 заморожен, L_orth всегда)
  #    грузит 3b_mof/moF_best.pt, сохраняет в checkpoints/3b_mof_stage2/
  !FSTNET_STAGE=2 FSTNET_DATA=data/jarvis_special.json FSTNET_EPOCHS=4 python train_colab_mof.py
  # 4) проверка
  !python eval_mof.py --ckpt checkpoints/3b_mof_stage2/best.pt --val --gen 10
"""
import os
import sys
import json
import math
import shutil
import subprocess
import time

import numpy as np

def log(msg): print(msg, flush=True)

sys.path.insert(0, os.getcwd())

STAGE = int(os.environ.get("FSTNET_STAGE", "1"))
if STAGE not in (1, 2):
    raise ValueError("FSTNET_STAGE должен быть 1 или 2")
log(f"STAGE={STAGE}")

from colab_drive import setup_checkpoint_dir
STAGE1_SUBDIR = "3b_mof"
CKPT_DIR = setup_checkpoint_dir(subdir=STAGE1_SUBDIR if STAGE == 1 else "3b_mof_stage2")
CONTENT = "/content" if os.path.isdir("/content") else CKPT_DIR
CKPT_DRIVE = os.path.join(CKPT_DIR, "moF_best.pt")
CKPT_LOCAL = os.path.join(CONTENT, f"best_3b_mof{'' if STAGE == 1 else '_stage2'}.pt")
FINAl_LOCAL = os.path.join(CONTENT, f"final_3b_mof{'' if STAGE == 1 else '_stage2'}.pt")
CACHE_DRIVE = os.path.join(CKPT_DIR, "jarvis_mof_samples.npz")
CACHE_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}.npz")


def pick_source(drive_path, local_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100_000_000:
        return local_path
    if os.path.exists(drive_path):
        log(f"  copy c Drive -> {local_path}")
        try:
            shutil.copyfile(drive_path, local_path)
            return local_path
        except Exception as e:
            log(f"  copy failed ({e}); читаю с Диска")
            return drive_path
    return None


log("Installing deps...")
subprocess.run(["pip", "install", "-q", "tokenizers", "tqdm",
                "--break-system-packages"],
               check=False)

import torch
import torch.nn as nn
import torch.amp
from torch.cuda.amp import GradScaler
autocast = torch.amp.autocast
from tqdm import tqdm

from config_3b_mof import FSTMoFConfig
from model.core_mof import FSTMoFModel
from tokenizers import Tokenizer

cfg = FSTMoFConfig()
for k in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "d_ff",
          "n_fields", "field_rank", "gating_top_k", "max_seq_len", "hidden_alpha"):
    v = os.environ.get(f"FSTNET_{k.upper()}", "").strip()
    if v:
        setattr(cfg, k, int(v))
        log(f"env override: {k}={v}")
model = FSTMoFModel(cfg)

start_step = 0
if STAGE == 2:
    stage1_dir = setup_checkpoint_dir(subdir=STAGE1_SUBDIR)
    src = pick_source(os.path.join(stage1_dir, "moF_best.pt"),
                      os.path.join(CONTENT, "best_3b_mof.pt"))
    if not src:
        log("[FAIL] Stage 2 требует чекпоинт Stage 1 (checkpoints/3b_mof/moF_best.pt). "
            "Сначала обучай Stage 1.")
        sys.exit(1)
else:
    src = pick_source(CKPT_DRIVE, CKPT_LOCAL)
if src:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    start_step = ck.get("step", 0)
    log(f"Resumed from {src} (step {start_step})")
else:
    log("Свежий старт (0).")

params = sum(p.numel() for p in model.parameters())
log(f"Params: {params/1e9:.3f}B | 1-bit storage: {params/8/1e6:.0f}MB")

tok = Tokenizer.from_file(cfg.tokenizer_path)
IM_S, IM_E = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100


def build_loss_mask(ids, roles):
    mask = [0] * len(ids)
    i = 0
    for role, content in roles:
        seg = tok.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
        if role == "assistant":
            for j in range(i, min(i + len(seg), len(ids))):
                mask[j] = 1
        i += len(seg)
    return mask


def make_samples(path, seq_len):
    data = json.load(open(path))
    xs, ys = [], []
    for conv in data:
        roles = [(r, c) for r, c in conv]
        ids = []
        for role, content in roles:
            ids += tok.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
        if len(ids) < 8 or len(ids) > seq_len:
            continue
        lm = build_loss_mask(ids, roles)
        x, y = [], []
        for i in range(seq_len):
            j = i + 1
            x.append(ids[i] if i < len(ids) else PAD)
            y.append(ids[j] if (j < len(ids) and lm[j]) else IGNORE)
        xs.append(x); ys.append(y)
    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


class DS(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x; self.y = y
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).long(), torch.from_numpy(self.y[i]).long()


log("Pre-tokenization...")
npz = pick_source(CACHE_DRIVE, CACHE_LOCAL)
if npz:
    d = np.load(npz, mmap_mode="r")
    X, Y = d["x"], d["y"]
    log(f"  cache: {npz}")
else:
    X, Y = make_samples(os.environ.get("FSTNET_DATA", "data/jarvis_full.json"), cfg.max_seq_len)
    np.savez_compressed(CACHE_DRIVE, x=X, y=Y)
    try:
        np.savez_compressed(CACHE_LOCAL, x=X, y=Y)
    except Exception:
        pass
    log(f"  tokenized -> {CACHE_DRIVE}")

rng_seed = int(os.environ.get("JARVIS_SEED", "42"))
np.random.seed(rng_seed)
perm = np.random.permutation(len(X))
X, Y = X[perm], Y[perm]
val_n = int(len(X) * 0.03)
val_x, val_y = X[:val_n], Y[:val_n]
tr_x, tr_y = X[val_n:], Y[val_n:]
log(f"Train: {len(tr_x)}, Val: {len(val_x)}")

train_ds = DS(tr_x, tr_y)
val_ds = DS(val_x, val_y)
BATCH = int(os.environ.get("FSTNET_BATCH", "8"))
ACCUM = int(os.environ.get("FSTNET_ACCUM", "8"))
WORKERS = int(os.environ.get("FSTNET_WORKERS", "0"))
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
    pin_memory=False, persistent_workers=WORKERS > 0, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
    pin_memory=False, persistent_workers=WORKERS > 0)

EPOCHS = int(os.environ.get("FSTNET_EPOCHS", "4"))
LEARN_RATE = float(os.environ.get("FSTNET_LR", "2e-4"))
step0 = start_step if STAGE == 1 else 0
total_steps = EPOCHS * len(tr_x) // (BATCH * ACCUM) + step0
warmup = max(total_steps // 100, 50)

all_names = [n for n, _ in model.named_parameters()]
W0_NAMES = [n for n in all_names if ".W0" in n]
FIELD_NAMES = [n for n in all_names if ".Fg." in n or ".Fu." in n or ".Fd." in n or ".hyper." in n]

opt_params = ([p for n, p in model.named_parameters() if n in FIELD_NAMES]
              if STAGE == 2 else list(model.parameters()))
opt = torch.optim.AdamW(opt_params, lr=LEARN_RATE, foreach=False, fused=True)

def lr_fn(s):
    if s < warmup: return (s + 1) / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))

sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
USE_BF16 = cap >= 8
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
log(f"GPU sm_{cap}; dtype={COMPUTE_DTYPE}")
scaler = None if (USE_BF16 or device == "cpu") else GradScaler()

if os.environ.get("FSTNET_COMPILE", "").strip() not in ("", "0"):
    try:
        if device == "cuda":
            model = torch.compile(model, dynamic=True)
            log("torch.compile(dynamic=True) ON")
    except Exception as e:
        log(f"compile skip: {e}")

def apply_phase(p, freeze_w0):
    ratio = 1.0 if STAGE == 2 else min(1.0, max(0.0, (p - 0.15) / 0.45))
    model.set_binarize(ratio)
    for name, d in model.named_parameters():
        if STAGE == 2:
            d.requires_grad_(name in FIELD_NAMES)
        else:
            d.requires_grad_(not freeze_w0 or name not in W0_NAMES)
    for m in model.modules():
        if hasattr(m, "quant_in"):
            m.quant_in = STAGE == 2 or p >= 0.5


SEQ = cfg.max_seq_len
step = step0
t0 = time.time()
best_val = float("inf")
apply_phase(step / max(1, total_steps), freeze_w0=False)
model.train()

log(f"Training: {total_steps} steps | batch={BATCH} accum={ACCUM} (eff {BATCH*ACCUM}) | "
    f"lr={LEARN_RATE:.1e} | EPOCHS={EPOCHS} | seq={SEQ} | fields={cfg.n_fields} topk={cfg.gating_top_k}")

for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}")
    opt.zero_grad(set_to_none=True)
    for bx, by in pbar:
        bx, by = (bx.to(device, non_blocking=True),
                  by.to(device, non_blocking=True))
        p = step / max(1, total_steps)
        freeze = p > 0.6
        apply_phase(p, freeze_w0=freeze)
        with autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            logits, ce = model(bx, by)
            ls = ce
            use_orth = STAGE == 2 or p > 0.6
            orth = model.orth_loss() if use_orth else None
            loss = ls + cfg.orth_scale * orth if use_orth else ls
            loss = loss / ACCUM
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step += 1
        if step % ACCUM == 0:
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sch.step()

        if step % 50 == 0:
            el = time.time() - t0
            eta = el / max(step - step0, 1) * (total_steps - step)
            curr_ratio = model.blocks[0].ffn.W0g.binarize
            log(f"Step {step}/{total_steps} | CE {ce.item():.4f}"
                f"{' ORTH ' + f'{orth.item():.4f}' if use_orth else ''} | "
                f"β {curr_ratio:.2f} | LR {sch.get_last_lr()[0]:.2e} | "
                f"VRAM {torch.cuda.memory_allocated()/1024**2:.0f}MB | ETA {eta/60:.0f}min")

        if step % 500 == 0:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                        _, l = model(vx, vy)
                    vl += l.item() * len(vx); vn += len(vx)
            val_avg = vl / max(vn, 1)
            log(f"  VAL CE: {val_avg:.4f}")
            if val_avg < best_val:
                best_val = val_avg
                state = {"step": step, "model_state": model.state_dict(), "config": cfg}
                torch.save(state, CKPT_LOCAL)
                shutil.copyfile(CKPT_LOCAL, CKPT_DRIVE)
                log(f"  >> best saved (val {best_val:.4f})")
            model.train()

state = {"step": step, "model_state": model.state_dict(), "config": cfg}
torch.save(state, FINAl_LOCAL)
shutil.copyfile(FINAl_LOCAL, os.path.join(CKPT_DIR, "final.pt"))
log(f"DONE. Step={step}, best_val={best_val:.4f}")
log(f"Следующее: 1-bit export (S3) + bitnet.cpp fork (см. SPEC_3B_MOF.md).")