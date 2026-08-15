#!/usr/bin/env python3
"""bit_field_tune.py — дообучение ContinuousField U/V под 1-bit (STE) до экспорта.

Честный 1-bit без потери перплексии невозможен постафактум: если просто
бинаризовать U/V (sign·s) на экспорте, веса уйдут из выученного
градиентного пространства. Решение — короткая дообучающая фаза, где поля
проходят СКВОЗЬ STE-binarize (как BitLinear в основном обучении): градиент
адаптирует U/V к их квантованным значениям sign(U)·s, ошибка экспорта -> 0.

Запуск (после S1/S2, на готовом кэше):
  FSTNET_TUNE_STEPS=300 FSTNET_LR=5e-5 python jarvis_engine/trainer/bit_field_tune.py

Что делает:
  - подхватывает последний чекпоинт (3b_mof/moF_best.pt или _stage2, env FSTNET_STAGE),
  - ставит binarize=1.0 для ВСЕХ BitLinear и ContinuousField,
  - обучает ТОЛЬКО поля+гиперсеть (как Stage 2), Adafactor, ~300 шагов,
  - сохраняет moF_best_1bit.pt (локально + на Диск) для export_1bit.py.
"""
import os
import sys
import json
import math
import time
import shutil
import threading
import zipfile
import io
import subprocess

if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import contextlib
from numpy.lib import format as npyfmt
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast  # torch>=2.3
    _AMP_DTYPE_KW = {"device_type": "cuda", "dtype": torch.float16}
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # torch<2.3
    _AMP_DTYPE_KW = {"dtype": torch.float16}

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.dirname(_THIS))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "brain"))
sys.path.insert(0, os.path.join(_ROOT, "brain", "model"))

from config_3b_mof import FSTMoFConfig
from model.core_mof import FSTMoFModel
from ste_optimizer import Adafactor
from colab_drive import setup_checkpoint_dir, mount_drive, save_checkpoint, load_checkpoint


def log(m): print(m, flush=True)


STAGE = int(os.environ.get("FSTNET_STAGE", "1"))
TUNE_STEPS = int(os.environ.get("FSTNET_TUNE_STEPS", "300"))
LEARN_RATE = float(os.environ.get("FSTNET_LR", "5e-5"))
BATCH = int(os.environ.get("FSTNET_BATCH", "2"))
ACCUM = int(os.environ.get("FSTNET_ACCUM", "32"))
SEQ = int(os.environ.get("FSTNET_SEQ", "512"))
WORKERS = int(os.environ.get("FSTNET_WORKERS", "0"))
SAMPLES = int(os.environ.get("FSTNET_SAMPLES", "100000"))
SEED = int(os.environ.get("JARVIS_SEED", "42"))
VAL_MAX = int(os.environ.get("FSTNET_VAL_MAX", "512"))
CONTENT = "/content" if os.path.isdir("/content") else "."

log(f"STAGE={STAGE} tune_steps={TUNE_STEPS} lr={LEARN_RATE:.1e} "
    f"batch={BATCH} accum={ACCUM} (eff {BATCH*ACCUM})")

device = "cuda" if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
COMPUTE_DTYPE = (torch.bfloat16 if cap >= 8 else torch.float16) if device == "cuda" else torch.float32
USE_SCALER = device == "cuda" and COMPUTE_DTYPE == torch.float16 and \
    os.environ.get("FSTNET_SCALER", "1") != "0"
scaler = GradScaler() if USE_SCALER else None
log(f"device={device} sm_{cap}; dtype={COMPUTE_DTYPE}"
    f" ({'GradScaler ON' if USE_SCALER else 'без GradScaler'})")

cfg = FSTMoFConfig()
for k in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "d_ff",
          "n_fields", "field_rank", "gating_top_k", "max_seq_len", "hidden_alpha"):
    v = os.environ.get(f"FSTNET_{k.upper()}", "").strip()
    if v:
        setattr(cfg, k, int(v))
        log(f"env override: {k}={v}")

STAGE1_SUBDIR = "3b_mof"
STAGE_SUBDIR = STAGE1_SUBDIR if STAGE == 1 else "3b_mof_stage2"
CKPT_DIR = setup_checkpoint_dir(subdir=STAGE_SUBDIR)
CKPT_DRIVE = os.path.join(CKPT_DIR, "moF_best_1bit.pt")
CKPT_LOCAL = os.path.join(CONTENT, "best_3b_mof_1bit.pt")

torch.set_default_dtype(COMPUTE_DTYPE)
model = FSTMoFModel(cfg).to(device=device)
torch.set_default_dtype(torch.float32)

# подхват последнего чекпоинта (эта же стадия)
SRC_SUBDIR = STAGE_SUBDIR
src_dir = setup_checkpoint_dir(subdir=SRC_SUBDIR)
src = os.path.join(src_dir, "moF_best.pt")
local_src = os.path.join(CONTENT, f"best_3b_mof{'' if STAGE == 1 else '_stage2'}.pt")
if not os.path.exists(src) and os.path.exists(local_src):
    src = local_src
if os.path.exists(src):
    ck = load_checkpoint(src)
    ms = ck["model_state"]
    if "head.scale" in ms and not getattr(cfg, "quant_head", False):
        log("[WARN] Старый BitLinear head -> убираю head.scale/head.bias")
        ms = {k: v for k, v in ms.items() if k not in ("head.scale", "head.bias")}
    model.load_state_dict(ms)
    log(f"Loaded {src} (step {ck.get('step', '?')})")
else:
    log(f"[FAIL] чекпоинт не найден: {src}")
    sys.exit(1)

params = sum(p.numel() for p in model.parameters())
log(f"Params: {params/1e9:.3f}B | 1-bit storage: {params/8/1e6:.0f}MB")

# ---- данные: готовый кэш (как run_trainer) ------------------------------
X_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_x.npy")
Y_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_y.npy")
CACHE_DRIVE_NPZ = os.path.join(CKPT_DIR, "jarvis_mof_samples.npz")
CACHE_LOCAL_NPZ = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}.npz")


def extract_npz_to_memmap(npz_path, x_out, y_out, chunk=16 << 20):
    import ast as _ast
    with zipfile.ZipFile(npz_path) as zf:
        for key, out in (("x", x_out), ("y", y_out)):
            with zf.open(f"{key}.npy") as s:
                if s.read(6) != b"\x93NUMPY":
                    raise ValueError(f"{npz_path}: {key}.npy не npy")
                ver = s.read(2)
                hlen = (int.from_bytes(s.read(2), "little") if ver == b"\x01\x00"
                        else int.from_bytes(s.read(4), "little"))
                meta = _ast.literal_eval(s.read(hlen).decode("latin-1"))
                dt = np.dtype(meta["descr"]); shape = tuple(meta["shape"])
                mm = npyfmt.open_memmap(out, mode="w+", dtype=dt, shape=shape)
                flat = mm.reshape(-1); isz = dt.itemsize; off = 0; tail = b""
                while True:
                    buf = s.read(chunk)
                    if not buf:
                        break
                    buf = tail + buf
                    n = len(buf) // isz
                    if n:
                        flat[off:off + n] = np.frombuffer(buf[:n * isz], dtype=dt)
                        off += n
                    tail = buf[n * isz:]
                mm.flush()
                log(f"  extracted {key} -> {out} ({mm.shape})")


def ensure_cache():
    if os.path.exists(X_LOCAL) and os.path.exists(Y_LOCAL):
        return np.load(X_LOCAL, mmap_mode="r"), np.load(Y_LOCAL, mmap_mode="r")
    for npz in (CACHE_LOCAL_NPZ, CACHE_DRIVE_NPZ):
        if os.path.exists(npz):
            if npz is CACHE_DRIVE_NPZ and "/content/drive" in npz:
                shutil.copyfile(npz, CACHE_LOCAL_NPZ)
                npz = CACHE_LOCAL_NPZ
            log(f"  npz -> memmap: {npz}")
            extract_npz_to_memmap(npz, X_LOCAL, Y_LOCAL)
            return np.load(X_LOCAL, mmap_mode="r"), np.load(Y_LOCAL, mmap_mode="r")
    return None, None


X, Y = ensure_cache()
if X is None:
    log("[FAIL] кэш данных не найден (запусти сначала run_trainer для токенизации)")
    sys.exit(1)

W = X.shape[1]
lens = np.empty(len(X), dtype=np.int64)
CH = 8192
for s in range(0, len(X), CH):
    b = np.asarray(X[s:s + CH]) != 0
    col = b[:, ::-1].argmax(axis=1)
    lens[s:s + CH] = np.where(b.any(axis=1), W - col, 0)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
val_n = max(int(len(X) * 0.03), 1)
val_idx = perm[:val_n]
tr_idx = perm[val_n:]
if len(tr_idx) > SAMPLES:
    tr_idx = tr_idx[:SAMPLES]
log(f"Train: {len(tr_idx)}, Val: {len(val_idx)}")


class DS(torch.utils.data.Dataset):
    def __init__(self, x, y, idx, seq_len, lens):
        self.x = x; self.y = y; self.idx = idx
        self.seq_len = seq_len; self.lens = lens
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        k = int(self.idx[i])
        L = int(self.lens[k])
        if L > self.seq_len:
            s = L - self.seq_len
            xs, ys = self.x[k][s:L], self.y[k][s:L]
        else:
            xs, ys = self.x[k][:self.seq_len], self.y[k][:self.seq_len]
        return (torch.from_numpy(np.ascontiguousarray(xs, dtype=np.int64).copy()),
                torch.from_numpy(np.ascontiguousarray(ys, dtype=np.int64).copy()))


train_ds = DS(X, Y, tr_idx, seq_len=SEQ, lens=lens)
val_ds = DS(X, Y, val_idx, seq_len=SEQ, lens=lens)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
    pin_memory=False, persistent_workers=WORKERS > 0, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
    pin_memory=False, persistent_workers=WORKERS > 0)

# ---- 1-bit режим: ВСЁ (BitLinear и поля) через STE ----------------------
model.set_binarize(1.0)
log("binarize=1.0 для всех BitLinear + ContinuousField (STE)")

all_names = [n for n, _ in model.named_parameters()]
FIELD_NAMES = [n for n in all_names if ".Fg." in n or ".Fu." in n or ".Fd." in n or ".hyper." in n]
opt_params = [p for n, p in model.named_parameters() if n in FIELD_NAMES]
frozen = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in opt_params)
log(f"Оптимизирую поля+гиперсеть: {sum(p.numel() for p in opt_params)/1e6:.0f}M параметров "
    f"(заморожено: {frozen/1e9:.2f}B)")

opt = Adafactor(opt_params, lr=LEARN_RATE, eps=(1e-30, 1e-3),
                clip_threshold=1.0, decay_rate=-0.8, beta1=None,
                weight_decay=0.0, scale_parameter=False,
                relative_step=False, warmup_init=False)

warmup = max(TUNE_STEPS // 20, 10)
total_steps = TUNE_STEPS


def lr_fn(s):
    if s < warmup: return (s + 1) / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
crit = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

# ---- цикл ----------------------------------------------------------------
step = 0
t0 = time.time()
best_val = float("inf")
mmc = torch.cuda.memory_allocated if device == "cuda" else (lambda: 0)
peak = 0
model.train()
log(f"Tuning {TUNE_STEPS} steps | batch={BATCH} accum={ACCUM} | lr={LEARN_RATE:.1e}")

total_batches = max(len(train_loader) // ACCUM, 1)
pbar = tqdm(train_loader, desc="TUNE", total=total_batches * ACCUM, unit="batch")
opt.zero_grad(set_to_none=True)
for it, (bx, by) in enumerate(pbar):
    bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
    with autocast(**_AMP_DTYPE_KW) if device == "cuda" else contextlib.nullcontext():
        logits, ce = model(bx, by)
        orth = model.orth_loss()
        loss = (ce + cfg.orth_scale * orth) / ACCUM
    loss = loss.float()
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    if (it + 1) % ACCUM == 0:
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sch.step()
        step += 1
        if device == "cuda":
            torch.cuda.empty_cache()
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024**2)
        if step % 25 == 0 or step == total_steps:
            el = time.time() - t0
            eta = el / step * (total_steps - step) if step else 0
            log(f"Step {step}/{total_steps} | CE {ce.item():.4f} | "
                f"ORTH {orth.item():.4f} | LR {sch.get_last_lr()[0]:.2e} | "
                f"VRAM peak {peak/1024:.1f}GB | ETA {eta/60:.0f}min")
        if step % 100 == 0 or step == total_steps:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for vi, (vx, vy) in enumerate(val_loader):
                    if vi * BATCH >= VAL_MAX:
                        break
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    _, l = model(vx, vy)
                    vl += l.item() * len(vx); vn += len(vx)
            val_avg = vl / max(vn, 1)
            log(f"  VAL CE: {val_avg:.4f} (n={vn})")
            if val_avg < best_val:
                best_val = val_avg
                state = {"step": step, "model_state": model.state_dict(), "config": cfg,
                         "quant": {"binarize": 1.0, "fields": True}}
                save_checkpoint(CKPT_LOCAL, state)
                if device == "cuda":
                    def _up():
                        try:
                            os.makedirs(os.path.dirname(CKPT_DRIVE), exist_ok=True)
                            shutil.copyfile(CKPT_LOCAL, CKPT_DRIVE + ".part")
                            os.replace(CKPT_DRIVE + ".part", CKPT_DRIVE)
                        except Exception as e:
                            log(f"[upload] fail {e}")
                    threading.Thread(target=_up, daemon=True).start()
                log(f"  >> best 1bit saved (val {best_val:.4f})")
            model.train()

save_checkpoint(CKPT_LOCAL, {"step": step, "model_state": model.state_dict(), "config": cfg,
             "quant": {"binarize": 1.0, "fields": True}})
try:
    os.makedirs(os.path.dirname(CKPT_DRIVE), exist_ok=True)
    shutil.copyfile(CKPT_LOCAL, CKPT_DRIVE)
    log(f"DONE. saved {CKPT_LOCAL} -> {CKPT_DRIVE}")
except Exception as e:
    log(f"[WARN] drive upload: {e}")
log(f"DONE. Step={step}, best_val={best_val:.4f}")
log("Далее: python jarvis_engine/trainer/export_1bit.py")
