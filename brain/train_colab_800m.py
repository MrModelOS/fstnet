#!/usr/bin/env python3
"""Обучение FST-Net 800M (JARVIS) в Colab.

Датасет: data/jarvis_full.json (см. build_jarvis_data.py) — формат [[role, content], ...].
Маска потерь: только ответы assistant (и user/system/инструмент игнорируются).

Оптимизации:
  * bf16 на Ampere+ (sm>=80), fp16+GradScaler на T4 (sm_75)
  * SDPA attention (в model.core), torch.compile опционально (FSTNET_COMPILE=1)
  * batch 16 + accum 2 (эффектив. 32), pin_memory, num_workers=2
  * пре-токенизация в .npz на локальном SSD /content + постоянный кэш на Google Диске
  * чекпоинты пишутся сначала в /content, затем дублируются на Диск

Запуск в Colab:
  from google.colab import drive; drive.mount('/content/drive')
  !git clone https://github.com/MrModelOS/fstnet  (или %cd fstnet && git pull)
  %cd fstnet
  !pip install -q transformers datasets tokenizers tqdm
  !python build_jarvis_data.py --count 60000
  !FSTNET_EPOCHS=5 FSTNET_LR=3e-4 python train_colab_800m.py
"""
import os
import sys
import json
import math
import time
import random
import subprocess

import numpy as np

def log(msg): print(msg, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colab_drive import setup_checkpoint_dir
CKPT_DIR = setup_checkpoint_dir(subdir="800m")
CKPT_DRIVE = os.path.join(CKPT_DIR, "best.pt")
CKPT_LOCAL = "/content/best_800m.pt"
FINAL_LOCAL = "/content/final_800m.pt"
CACHE_DRIVE = os.path.join(CKPT_DIR, "jarvis_samples.npz")
CACHE_LOCAL = "/content/jarvis_samples.npz"


def pick_source(drive_path, local_path):
    """Возвращает путь: локальный SSD -> копия с Диска -> (нет)."""
    import shutil
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100_000_000:
        log(f"  local: {local_path}")
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
subprocess.run(["pip", "install", "-q", "tokenizers", "tqdm"], check=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp
from torch.cuda.amp import GradScaler
autocast = torch.amp.autocast
from tqdm import tqdm

from model.core import FSTNetCore
from config_800m import FSTConfig800M
from tokenizers import Tokenizer

cfg = FSTConfig800M()
model = FSTNetCore(cfg)

# resume (если есть чекпоинт)
src = pick_source(CKPT_DRIVE, CKPT_LOCAL)
start_step = 0
if src:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    model.load_checkpoint_into(ck["model_state"])
    start_step = ck.get("step", 0)
    log(f"Resumed from {src} (step {start_step})")
else:
    log("Свежий старт (0): чекпоинта нет.")

params = sum(p.numel() for p in model.parameters())
log(f"Params: {params/1e6:.1f}M ({params/1e9:.2f}B)")

tok = Tokenizer.from_file(cfg.tokenizer_path)
IM_S, IM_E = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100


def build_loss_mask(ids, roles):
    """Маска: тренируемся только на тексте роли 'assistant'.

    Возвращает список 0/1 длины len(ids) — какие токены предсказывать (shift позже).
    """
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
    d = np.load(npz)
    X, Y = d["x"], d["y"]
    log(f"  cache: {npz}")
else:
    X, Y = make_samples("data/jarvis_full.json", cfg.max_seq_len)
    np.savez_compressed(CACHE_DRIVE, x=X, y=Y)
    try:
        np.savez_compressed(CACHE_LOCAL, x=X, y=Y)
    except Exception:
        pass
    log(f"  tokenized -> {CACHE_DRIVE}")

rng_seed = int(os.environ.get("JARVIS_SEED", "42"))
random.seed(rng_seed)
perm = np.random.permutation(len(X))
X, Y = X[perm], Y[perm]
val_n = int(len(X) * 0.03)
val_x, val_y = X[:val_n], Y[:val_n]
tr_x, tr_y = X[val_n:], Y[val_n:]
log(f"Train: {len(tr_x)}, Val: {len(val_x)}")

train_ds = DS(tr_x, tr_y)
val_ds = DS(val_x, val_y)
BATCH = 16
ACCUM = 2
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2,
                          pin_memory=True, persistent_workers=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)

EPOCHS = int(os.environ.get("FSTNET_EPOCHS", "5"))
LEARN_RATE = float(os.environ.get("FSTNET_LR", "3e-4"))
total_steps = EPOCHS * len(tr_x) // (BATCH * ACCUM) + start_step
warmup = max(total_steps // 10, 1)

opt = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE, foreach=False, fused=True)

def lr_fn(s):
    if s < warmup: return (s + 1) / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))

sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

device = "cuda"
model = model.to(device)
cap = torch.cuda.get_device_capability(torch.cuda.current_device())[0] if torch.cuda.is_available() else 0
USE_BF16 = cap >= 8
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
log(f"GPU sm_{cap}; dtype={COMPUTE_DTYPE}")
scaler = None if USE_BF16 else GradScaler()

if os.environ.get("FSTNET_COMPILE", "").strip() not in ("", "0"):
    try:
        model = torch.compile(model, dynamic=True)
        log("torch.compile(dynamic=True) ON")
    except Exception as e:
        log(f"compile skip: {e}")

SEQ = cfg.max_seq_len
model.train()
step = start_step
t0 = time.time()
best_val = float("inf")

log(f"Training: {total_steps} steps | batch={BATCH} accum={ACCUM} (eff {BATCH*ACCUM}) | "
    f"lr={LEARN_RATE:.1e} | EPOCHS={EPOCHS} | seq={SEQ} | dtype={COMPUTE_DTYPE}")

for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}")
    opt.zero_grad(set_to_none=True)
    for bx, by in pbar:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            h, _ = model(bx, target_cycles=4, return_hidden=True)
            ls = torch.tensor(0.0, device=device)
            nv = 0
            for s in range(0, SEQ, 64):
                e = min(s + 64, SEQ)
                l = crit(model.head(h[:, s:e]).view(-1, cfg.vocab_size), by[:, s:e].reshape(-1))
                ls += l
                nv += (by[:, s:e] != IGNORE).sum().item()
            loss = ls / max(nv, 1) / ACCUM

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
            eta = el / max(step - start_step, 1) * (total_steps - step)
            log(f"Step {step}/{total_steps} | Loss {ls.item()/max(nv,1):.4f} | "
                f"LR {sch.get_last_lr()[0]:.2e} | VRAM {torch.cuda.memory_allocated()/1024**2:.0f}MB | "
                f"ETA {eta/60:.0f}min")

        if step % 250 == 0:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                        h, _ = model(vx, target_cycles=4, return_hidden=True)
                        for s in range(0, SEQ, 64):
                            e = min(s + 64, SEQ)
                            l = crit(model.head(h[:, s:e]).view(-1, cfg.vocab_size), vy[:, s:e].reshape(-1))
                            vl += l.item()
                            vn += (vy[:, s:e] != IGNORE).sum().item()
            val_avg = vl / max(vn, 1)
            log(f"  VAL LOSS: {val_avg:.4f}")
            if val_avg < best_val:
                best_val = val_avg
                state = {"step": step, "model_state": model.state_dict(), "config": cfg}
                torch.save(state, CKPT_LOCAL)
                import shutil
                shutil.copyfile(CKPT_LOCAL, CKPT_DRIVE)
                log(f"  >> best saved (val {best_val:.4f})")
            model.train()

import shutil
state = {"step": step, "model_state": model.state_dict(), "config": cfg}
torch.save(state, FINAL_LOCAL)
shutil.copyfile(FINAL_LOCAL, os.path.join(CKPT_DIR, "final.pt"))
log(f"DONE. Final step={step}, best_val={best_val:.4f}")
log(f"Activate JARVIS: конвертируй {CKPT_DRIVE} в GGUF Q8_0 и загрузи в Ollama.")