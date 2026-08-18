#!/usr/bin/env python3
"""Обучение FST-Net 1B 1-bit MoF — ускоренный тренер для Colab.

Архитектура: brain/config_1b_mof.py (~970M params, 24 слоя, dim=1280).
Оптимизации:
  - Большой батч (16 micro × 4 accum = 64 effective) вместо 1 × 32
  - Нет gradient checkpointing (1B влезает в VRAM с запасом)
  - Датасет целиком в RAM (1GB JSON → ~2GB RAM)
  - Pre-tokenization в RAM (не на диск)
  - Loss scaling x4096 вместо GradScaler (fp16-параметры)

Запуск в Colab:
  !python train_1b.py

Env:
  FSTNET_DATA      — путь к JSON (по умолчанию data/jarvis_full.json)
  FSTNET_BATCH     — micro-batch (16)
  FSTNET_ACCUM     — accumulation steps (4, effective=64)
  FSTNET_LR        — learning rate (3e-4)
  FSTNET_EPOCHS    — epochs (2)
  FSTNET_SEQ       — max seq len (2048)
  FSTNET_STEPS     — total steps (None=авто по датасету)
  FSTNET_DRIVE_SUB — подпапка на Google Диске (fstnet_1b)
"""
import os
import sys
import json
import math
import time
import contextlib

# Обязательно ДО инициализации CUDA: снижает фрагментацию.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "brain"))
sys.path.insert(0, os.path.join(_THIS, "jarvis_engine", "trainer"))

from config_1b_mof import FSTMoFConfig
from model.core_mof import FSTMoFModel
from ste_optimizer import Adafactor

try:
    from torch.amp import autocast
    _AMP_DTYPE_KW = {"device_type": "cuda", "dtype": torch.float16}
except ImportError:
    from torch.cuda.amp import autocast
    _AMP_DTYPE_KW = {"dtype": torch.float16}


def log(msg):
    print(msg, flush=True)


# ─── Env ──────────────────────────────────────────────────────────────
DATA_PATH   = os.environ.get("FSTNET_DATA", "data/jarvis_full.json")
BATCH       = int(os.environ.get("FSTNET_BATCH", "16"))
ACCUM       = int(os.environ.get("FSTNET_ACCUM", "4"))
LR          = float(os.environ.get("FSTNET_LR", "3e-4"))
EPOCHS      = int(os.environ.get("FSTNET_EPOCHS", "2"))
SEQ_LEN     = int(os.environ.get("FSTNET_SEQ", "2048"))
TOTAL_STEPS = int(os.environ["FSTNET_STEPS"]) if "FSTNET_STEPS" in os.environ else None
DRIVE_SUB   = os.environ.get("FSTNET_DRIVE_SUB", "fstnet_1b")
CKPT_DIR    = os.environ.get("FSTNET_CKPT_DIR", "")

LOSS_SCALE  = 4096.0  # fp16 loss scaling (exact ÷2^k)

# ─── Paths ────────────────────────────────────────────────────────────
CONTENT     = "/content" if os.path.isdir("/content") else os.path.expanduser("~")
CKPT_LOCAL  = os.path.join(CONTENT, "best_1b_mof.pt")
CKPT_FINAL  = os.path.join(CONTENT, "final_1b_mof.pt")
CKPT_STEP   = os.path.join(CONTENT, "ckpt_step")  # каждые 1000 шагов

# Google Drive
DRIVE_PATH  = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else ""
CKPT_DRIVE  = os.path.join(DRIVE_PATH, DRIVE_SUB, "best_1b_mof.pt") if DRIVE_PATH else ""

# ─── Tokenizer ────────────────────────────────────────────────────────
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(
            os.path.join(_THIS, "brain", "tokenizer", "fst_bpe.json"))
    return _tokenizer


PAD_ID   = 0
IGNORE_ID = -100
IM_S = "<|im_start|>"
IM_E = "<|im_end|>"


def encode_conv(conv):
    """Кодирует диалог (role, content кортежи) с маской loss на ответах."""
    tok = get_tokenizer()
    ids, bounds = [], []
    for role, content in conv:
        seg = tok.encode(IM_S + role + "\n" + content + IM_E).ids
        bounds.append((len(ids), len(ids) + len(seg), role))
        ids += seg
    return ids, bounds


def loss_mask(ids, bounds):
    mask = [0] * len(ids)
    for s, e, role in bounds:
        if role == "assistant":
            for j in range(s, e):
                mask[j] = 1
    return mask


# ─── Dataset ──────────────────────────────────────────────────────────
MMAP_X = os.path.join(_THIS, "data", "x_1b.mmap")
MMAP_Y = os.path.join(_THIS, "data", "y_1b.mmap")
MMAP_L = os.path.join(_THIS, "data", "lens_1b.npy")


class ChatDS(Dataset):
    def __init__(self, path, seq_len, subsample=None):
        log(f"Loading dataset: {path}")
        data = json.load(open(path))
        N_raw = len(data)
        log(f"  raw conversations: {N_raw}")

        # Single pass: stream straight into memmap, nothing kept in RAM.
        # (rows in Python lists = 10+ GB → OOM; only one row at a time here.)
        mmx = np.lib.format.open_memmap(MMAP_X, mode="w+",
                                        dtype=np.int32, shape=(N_raw, seq_len))
        mmy = np.lib.format.open_memmap(MMAP_Y, mode="w+",
                                        dtype=np.int32, shape=(N_raw, seq_len))
        lens_arr = np.empty(N_raw, dtype=np.int64)

        n = 0
        import gc
        for i, conv in enumerate(data):
            ids, bounds = encode_conv(conv)
            L = len(ids)
            if L < 8:
                continue
            lm = loss_mask(ids, bounds)
            lens_arr[n] = L
            if L > seq_len:
                ids = ids[L - seq_len:]
                lm = lm[L - seq_len:]
                L = seq_len
            x = np.full(seq_len, PAD_ID, dtype=np.int32)
            y = np.full(seq_len, IGNORE_ID, dtype=np.int32)
            x[:L] = ids
            # loss only on assistant tokens
            for j in range(1, L):
                if lm[j]:
                    y[j - 1] = ids[j]
            mmx[n] = x
            mmy[n] = y
            n += 1
            if i % 50000 == 0:
                gc.collect()
        del data
        gc.collect()

        mmx.flush()
        mmy.flush()
        log(f"  valid samples: {n}")

        if subsample and n > subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, subsample, replace=False)
            mmx = mmx[idx]
            mmy = mmy[idx]
            lens_arr = lens_arr[idx]
            n = len(idx)
            log(f"  subsampled to: {n}")

        self.mx = mmx
        self.my = mmy
        self.lens = lens_arr[:n]
        np.save(MMAP_L, self.lens)
        log(f"  memmap written: {MMAP_X}, {MMAP_Y}")

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        # Хвост длинных диалогов уже записан в начало строки при построении.
        # torch.tensor(): свежая резервируемая память (from_numpy на memmap
        # даёт не-resizable storage -> collate падает с resize_).
        return (torch.tensor(self.mx[i], dtype=torch.long),
                torch.tensor(self.my[i], dtype=torch.long))


# ─── Checkpoint ───────────────────────────────────────────────────────
def save_ckpt(path, state):
    """Сохраняет чекпоинт: pack=True если модель полностью бинаризована."""
    from colab_drive import save_checkpoint
    save_checkpoint(path, state, pack=True)


def load_ckpt(path):
    """Загружает чекпоинт: авто-распаковка знаков."""
    from colab_drive import load_checkpoint
    return load_checkpoint(path)


def save_step_ckpt(tag):
    """Чекпоинт каждые 1000 шагов: /content/ckpt_step/step_XXXX.pt + на Drive."""
    os.makedirs(CKPT_STEP, exist_ok=True)
    st = {
        "step": step,
        "model_state": model.state_dict(),
        "config": cfg,
        "best_val": best_val,
    }
    p = os.path.join(CKPT_STEP, f"step_{step:06d}_{tag}.pt")
    save_ckpt(p, st)
    log(f"  [CKPT] saved {p}")
    if CKPT_DRIVE:
        import shutil
        try:
            shutil.copy2(p, os.path.join(os.path.dirname(CKPT_DRIVE), os.path.basename(p)))
        except Exception as e:
            log(f"  [WARN] drive copy failed: {e}")


# ─── Model ────────────────────────────────────────────────────────────
cfg = FSTMoFConfig(max_seq_len=SEQ_LEN)
device = "cuda" if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0

if device == "cuda":
    COMPUTE_DTYPE = torch.bfloat16 if cap >= 8 else torch.float16
else:
    COMPUTE_DTYPE = torch.float32

log(f"device={device} sm_{cap}; dtype={COMPUTE_DTYPE}; batch={BATCH} accum={ACCUM} (eff={BATCH*ACCUM})")

torch.set_default_dtype(COMPUTE_DTYPE)
model = FSTMoFModel(cfg).to(device)
torch.set_default_dtype(torch.float32)

# ─── Resume ───────────────────────────────────────────────────────────
start_step = 0
best_val = float("inf")

if os.path.exists(CKPT_LOCAL):
    log(f"Resuming from {CKPT_LOCAL}")
    ck = load_ckpt(CKPT_LOCAL)
    ms = ck["model_state"]
    if "head.scale" in ms:
        del ms["head.scale"]
    if "head.bias" in ms:
        del ms["head.bias"]
    model.load_state_dict(ms, strict=False)
    start_step = ck.get("step", 0)
    best_val = ck.get("best_val", float("inf"))
    log(f"  step={start_step}, best_val={best_val:.4f}")

log(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M "
    f"| 1-bit storage: {cfg.bytes_1bit()/1e6:.0f}MB")

# torch.compile: kernel fusion. НЕ reduce-overhead — CUDA graphs держат
# все 1900+ промежуточных буферов в VRAM (OOM на T4 16GB). Обычный mode.
from memory_manager import enable_if_env
mm = enable_if_env(device=device)          # grad ckpt по умолчанию ВКЛ
mm.wrap_model(model)
if device == "cuda":
    _COMPILE = os.environ.get("FSTNET_COMPILE", "1").strip()
    if _COMPILE in ("1", "true", "yes", "default"):
        log("Compiling model (torch.compile default)...")
        model = torch.compile(model, mode="default")
        log("  compile done")
    elif _COMPILE not in ("0", "false", "no"):
        log(f"Compiling model (torch.compile {_COMPILE})...")
        model = torch.compile(model, mode=_COMPILE)
        log("  compile done")
    else:
        log("torch.compile disabled (FSTNET_COMPILE=0)")

# ─── Optimizer ────────────────────────────────────────────────────────
opt_params = [p for p in model.parameters() if p.requires_grad]
opt = Adafactor(opt_params, lr=LR, eps=(1e-30, 1e-3),
                clip_threshold=1.0, decay_rate=-0.8,
                weight_decay=0.0, relative_step=False,
                scale_parameter=True, warmup_init=False)
sch = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, pct_start=0.05, div_factor=25,
    final_div_factor=100, total_steps=TOTAL_STEPS or 1,
    cycle_momentum=False)  # Adafactor не имеет momentum/betas

log(f"Optimizer: Adafactor (lr={LR}, OneCycle)")

# ─── Data ─────────────────────────────────────────────────────────────
# Должно быть ДО OneCycleLR: TOTAL_STEPS нужен для total_steps планировщика.
ds = ChatDS(DATA_PATH, SEQ_LEN)
train_loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True,
                          persistent_workers=True)

steps_per_epoch = max(len(train_loader) // ACCUM, 1)
if TOTAL_STEPS is None:
    TOTAL_STEPS = steps_per_epoch * EPOCHS
log(f"  dataset samples: {len(ds)} | steps/epoch: {steps_per_epoch}")
log(f"  epochs: {EPOCHS} | total steps: {TOTAL_STEPS} "
    f"| ckpt: every 1000 steps + end of epoch")

# ─── Binarize schedule ───────────────────────────────────────────────
def apply_phase(p):
    if p < 0.15:
        ratio = 0.0
    elif p < 0.6:
        ratio = (p - 0.15) / 0.45
    else:
        ratio = 1.0
    model.set_binarize(ratio)


# ─── Training ─────────────────────────────────────────────────────────
log(f"\nTraining: {TOTAL_STEPS} steps | batch={BATCH} accum={ACCUM} | lr={LR:.1e} | epochs={EPOCHS}")
log(f"  loss-scale x{LOSS_SCALE:.0f} (fp16 STE underflow protection)")

model.train()
step = start_step
step0 = step
t0 = time.time()
last_pulse = t0

mm.reset()
opt.zero_grad(set_to_none=True)

for epoch in range(EPOCHS):
    log(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    for it, (bx, by) in enumerate(train_loader):
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        p = step / max(1, TOTAL_STEPS)
        apply_phase(p)

        with autocast(**_AMP_DTYPE_KW) if device == "cuda" else contextlib.nullcontext():
            logits, ce = model(bx, by)
            use_orth = p > 0.6
            orth = model.orth_loss() if use_orth else None
            loss = ce + cfg.orth_scale * orth if use_orth else ce

        loss = loss.float() / ACCUM
        loss = loss * LOSS_SCALE
        mm.before_backward()
        loss.backward()

        if (it + 1) % ACCUM == 0:
            for pg in model.parameters():
                if pg.grad is not None:
                    pg.grad.div_(LOSS_SCALE)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sch.step()
            step += 1
            mm.after_step()

            if step % 1000 == 0:
                save_step_ckpt("periodic")

            if step % 50 == 0:
                now = time.time()
                eta = (now - t0) / max(step - step0, 1) * (TOTAL_STEPS - step)
                log(f"  step {step}/{TOTAL_STEPS} | CE {ce.item():.4f} | "
                    f"lr {sch.get_last_lr()[0]:.2e} | ETA {eta/60:.0f}min")

            if step >= TOTAL_STEPS:
                break

    save_step_ckpt("epoch_end")
    if step >= TOTAL_STEPS:
        break

# ─── Validation ───────────────────────────────────────────────────────
log("\nRunning validation...")
model.eval()
val_losses = []
with torch.no_grad():
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        logits, ce = model(bx, by)
        val_losses.append(ce.item())
        if len(val_losses) >= 100:
            break
val_avg = np.mean(val_losses)
log(f"  val CE: {val_avg:.4f} (n={len(val_losses)})")

# ─── Save final ───────────────────────────────────────────────────────
state = {
    "step": step,
    "model_state": model.state_dict(),
    "config": cfg,
    "best_val": best_val,
}

save_ckpt(CKPT_FINAL, state)
log(f"Saved final: {CKPT_FINAL}")

if val_avg < best_val:
    best_val = val_avg
    save_ckpt(CKPT_LOCAL, state)
    log(f"Saved best: {CKPT_LOCAL} (val={best_val:.4f})")

# Upload to Drive
if CKPT_DRIVE:
    os.makedirs(os.path.dirname(CKPT_DRIVE), exist_ok=True)
    import shutil
    shutil.copy2(CKPT_LOCAL, CKPT_DRIVE)
    log(f"Uploaded to Drive: {CKPT_DRIVE}")

log(f"\nDONE. Steps={step}, best_val={best_val:.4f}")
log(mm.report())
