#!/usr/bin/env python3
"""Обучение FST-Net 3B 1-bit MoF (JARVIS Core) в Colab.

Архитектура: model/core_mof.py (BitLinear+STE, ContinuousField, GQA+RoPE, гиперсеть).
Спецификация: SPEC_3B_MOF.md.

Оптимизации: модель сразу в fp16 (T4) / bf16 (Ampere+) без GradScaler;
токенизация в memmap (диск, без копий в RAM); оптимизатор Adafactor
(факторизованные состояния — не влезают AdamW-состояния 3B).

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
import zipfile
import io
from numpy.lib import format as npyfmt
from tqdm import tqdm

from config_3b_mof import FSTMoFConfig
from model.core_mof import FSTMoFModel
from tokenizers import Tokenizer


class Adafactor(torch.optim.Optimizer):
    """Adafactor (Shazeer & Stern 2018): адаптивные шаги с факторизованной
    памятью второго момента (~мегабайты вместо ~27GB AdamW на 3B).

    Внешний LR (scale_parameter=False, relative_step=False) — как в T5:
    каждый параметр нормализуется по RMS, LR задаёт RMS шага.

    Для fp16/bf16 весов внутри шага ведётся transient fp32 master-копия
    параметра (освобождается после обработки каждого параметра).
    """
    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0,
                 scale_parameter=True, relative_step=True, warmup_init=False):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True`")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1,
                        weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                state = self.state[p]
                grad_shape = grad.shape
                factored, use_first_moment = self._get_options(group, grad_shape)
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * lr)

                p_data_fp32.add_(-update)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
        return loss


def save_npz_streaming(path, arrays, chunk=4 << 20):
    """npz-бэкап без полной копии массивов в RAM: читает memmap и жмёт по кускам."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for key, arr in arrays:
            arr = np.ascontiguousarray(arr)
            hdr = io.BytesIO()
            npyfmt.write_array_header_1_0(hdr, npyfmt.header_data_from_array_1_0(arr))
            with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
                f.write(hdr.getvalue())
                mv = memoryview(arr).cast("B")
                for i in range(0, len(mv), chunk):
                    f.write(mv[i:i + chunk])


cfg = FSTMoFConfig()
for k in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "d_ff",
          "n_fields", "field_rank", "gating_top_k", "max_seq_len", "hidden_alpha"):
    v = os.environ.get(f"FSTNET_{k.upper()}", "").strip()
    if v:
        setattr(cfg, k, int(v))
        log(f"env override: {k}={v}")

device = "cuda" if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
if device == "cuda":
    COMPUTE_DTYPE = torch.bfloat16 if cap >= 8 else torch.float16
else:
    COMPUTE_DTYPE = torch.float32
log(f"device={device} sm_{cap}; модель в {COMPUTE_DTYPE} (Adafactor, без GradScaler)")
model = FSTMoFModel(cfg).to(device=device, dtype=COMPUTE_DTYPE)

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


def encode_conv(conv):
    ids, bounds = [], []
    for role, content in conv:
        seg = tok.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
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


def tokenize_to_memmap(path, seq_len, x_path, y_path):
    """Токенизация сразу в memmap (на диск), без сборки списков в RAM.

    Двухпроходная: счёт валидных сэмплов, затем заполнение строк.
    """
    data = json.load(open(path))
    count = 0
    for conv in data:
        ids, _ = encode_conv(conv)
        if 8 <= len(ids) <= seq_len:
            count += 1
    if count == 0:
        raise SystemExit(f"[FAIL] Нет валидных сэмплов в {path}")
    log(f"  валидных сэмплов: {count} / {len(data)}")
    xm = npyfmt.open_memmap(x_path, mode="w+", dtype=np.int32, shape=(count, seq_len))
    ym = npyfmt.open_memmap(y_path, mode="w+", dtype=np.int32, shape=(count, seq_len))
    i = 0
    for conv in data:
        ids, bounds = encode_conv(conv)
        L = len(ids)
        if L < 8 or L > seq_len:
            continue
        lm = np.asarray(loss_mask(ids, bounds), dtype=bool)
        xrow = np.full(seq_len, PAD, dtype=np.int32)
        xrow[:L] = ids
        yrow = np.full(seq_len, IGNORE, dtype=np.int32)
        if L > 1:
            yrow[:L - 1] = np.where(lm[1:], ids[1:], IGNORE)
        xm[i] = xrow
        ym[i] = yrow
        i += 1
        if i % 20000 == 0:
            log(f"  tokenized {i}/{count}")
    xm.flush(); ym.flush()
    del xm, ym


class DS(torch.utils.data.Dataset):
    def __init__(self, x, y, idx):
        self.x = x; self.y = y; self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        k = int(self.idx[i])
        return (torch.from_numpy(np.ascontiguousarray(self.x[k])).long(),
                torch.from_numpy(np.ascontiguousarray(self.y[k])).long())


log("Pre-tokenization...")
X_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_x.npy")
Y_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_y.npy")
if os.path.exists(X_LOCAL) and os.path.exists(Y_LOCAL):
    X = np.load(X_LOCAL, mmap_mode="r")
    Y = np.load(Y_LOCAL, mmap_mode="r")
    log(f"  memmap cache: {X_LOCAL}")
else:
    npz = pick_source(CACHE_DRIVE, CACHE_LOCAL)
    if npz:
        d = np.load(npz, mmap_mode="r")
        X, Y = d["x"], d["y"]
        log(f"  npz cache: {npz}")
    else:
        tokenize_to_memmap(os.environ.get("FSTNET_DATA", "data/jarvis_full.json"),
                           cfg.max_seq_len, X_LOCAL, Y_LOCAL)
        X = np.load(X_LOCAL, mmap_mode="r")
        Y = np.load(Y_LOCAL, mmap_mode="r")
        for dst in (CACHE_LOCAL, CACHE_DRIVE):
            try:
                save_npz_streaming(dst, (("x", X), ("y", Y)))
                log(f"  npz backup -> {dst}")
            except Exception as e:
                log(f"  [WARN] npz {dst} не записан: {e}")

rng_seed = int(os.environ.get("JARVIS_SEED", "42"))
rng = np.random.default_rng(rng_seed)
perm = rng.permutation(len(X))
val_n = max(int(len(X) * 0.03), 1)
val_idx = perm[:val_n]
tr_idx = perm[val_n:]
log(f"Train: {len(tr_idx)}, Val: {len(val_idx)}")

train_ds = DS(X, Y, tr_idx)
val_ds = DS(X, Y, val_idx)
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
total_steps = EPOCHS * len(tr_idx) // (BATCH * ACCUM) + step0
warmup = max(total_steps // 100, 50)

all_names = [n for n, _ in model.named_parameters()]
W0_NAMES = [n for n in all_names if ".W0" in n]
FIELD_NAMES = [n for n in all_names if ".Fg." in n or ".Fu." in n or ".Fd." in n or ".hyper." in n]

opt_params = ([p for n, p in model.named_parameters() if n in FIELD_NAMES]
              if STAGE == 2 else list(model.parameters()))
opt = Adafactor(opt_params, lr=LEARN_RATE, eps=(1e-30, 1e-3),
                clip_threshold=1.0, decay_rate=-0.8, beta1=None,
                weight_decay=0.0, scale_parameter=False,
                relative_step=False, warmup_init=False)
log("Optimizer: Adafactor (факторизованные состояния; клип через clip_threshold)")

def lr_fn(s):
    if s < warmup: return (s + 1) / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))

sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

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
        logits, ce = model(bx, by)
        ls = ce
        use_orth = STAGE == 2 or p > 0.6
        orth = model.orth_loss() if use_orth else None
        loss = ls + cfg.orth_scale * orth if use_orth else ls
        loss = loss / ACCUM
        loss.backward()
        step += 1
        if step % ACCUM == 0:
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