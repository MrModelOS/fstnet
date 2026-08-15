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
  !FSTNET_EPOCHS=4 FSTNET_LR=2e-4 python jarvis_engine/trainer/run_trainer.py
  # 3) обучение Stage 2 (спец. датасет, W0 заморожен, L_orth всегда)
  #    грузит 3b_mof/moF_best.pt, сохраняет в checkpoints/3b_mof_stage2/
  !FSTNET_STAGE=2 FSTNET_DATA=data/jarvis_special.json FSTNET_EPOCHS=4 python jarvis_engine/trainer/run_trainer.py
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
import threading

# Обязательно ДО инициализации CUDA-аллокатора: снижает фрагментацию (T4 16GB).
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np

def log(msg): print(msg, flush=True)

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)                       # ste_optimizer, memory_manager (этот каталог)
sys.path.insert(0, os.path.dirname(_THIS))      # jarvis_engine/
_ROOT = os.path.dirname(os.path.dirname(_THIS)) # корень fstnet/
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "brain"))  # config_3b_mof, model, colab_drive

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
CACHE_LOCAL_NPZ = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}.npz")
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


class AsyncUploader:
    """Копирует чекпоинт на Диск в фоне, не блокируя тренинг.
    Атомарно через .part -> os.replace; latest-wins (старый upload отменяется)."""

    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()
        self._job = None

    def submit(self, local, drive):
        with self._lock:
            self._job = (local, drive)
        if self._thread is None or not self._thread.is_alive():
            t = threading.Thread(target=self._run, daemon=True)
            self._thread = t
            t.start()

    def _run(self):
        while True:
            with self._lock:
                job = self._job
                self._job = None
            if job is None:
                return
            local, drive = job
            try:
                part = drive + ".part"
                shutil.copyfile(local, part)
                os.replace(part, drive)
            except Exception as e:
                log(f"[upload] {drive}: fail {e}")
                continue


uploader = AsyncUploader()


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

from ste_optimizer import Adafactor
from memory_manager import MemoryManager, enable_if_env


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


def extract_npz_to_memmap(npz_path, x_out, y_out):
    """npz -> два .npy memmap потоково (без распаковки массива в RAM).
    Для сжатых npz np.load(mmap_mode='r') НЕ даёт memmap — распаковывает всё
    в память (X+Y = 16GB при ~485k seq 4096 = OOM). Здесь читаем headers и
    пишем raw-данные напрямую в memmap-файлы порциями через ZipExtFile.read.

    RSS-ом это не взрывает: распаковка идёт порциями, в RAM держится только
    текущий чанк (CHUNK), а не весь массив.
    """
    import ast as _ast
    CHUNK = 16 << 20  # 16MB на чанк декомпрессии
    with zipfile.ZipFile(npz_path) as zf:
        for key, out in (("x", x_out), ("y", y_out)):
            with zf.open(f"{key}.npy") as src:
                magic = src.read(6)
                if magic != b"\x93NUMPY":
                    raise ValueError(f"{npz_path}: {key}.npy не npy-формат")
                ver = src.read(2)
                if ver == b"\x01\x00":
                    hlen = int.from_bytes(src.read(2), "little")
                elif ver == b"\x02\x00":
                    hlen = int.from_bytes(src.read(4), "little")
                else:
                    raise ValueError(f"{npz_path}: версия npy {ver!r} не поддерживается")
                hdr = src.read(hlen)
                meta = _ast.literal_eval(hdr.decode("latin-1"))
                dt = np.dtype(meta["descr"])
                shape = tuple(meta["shape"])
                mm = npyfmt.open_memmap(out, mode="w+", dtype=dt, shape=shape)
                flat = mm.reshape(-1)
                itemsize = dt.itemsize
                off = 0
                tail = b""
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    buf = tail + buf
                    n = len(buf) // itemsize
                    if n:
                        flat[off:off + n] = np.frombuffer(buf[:n * itemsize], dtype=dt)
                        off += n
                    tail = buf[n * itemsize:]
                mm.flush()
                log(f"  extracted {key} -> {out} ({tuple(mm.shape)}, "
                    f"{mm.nbytes / 1e9:.1f}GB)")
                del flat, mm


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
torch.set_default_dtype(COMPUTE_DTYPE)  # создаём сразу в fp16/bf16: 3.4B fp32 на CPU = OOM (~13.6GB)
model = FSTMoFModel(cfg).to(device=device)
torch.set_default_dtype(torch.float32)

mm = enable_if_env(device)
mm.wrap_model(model)

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
    def __init__(self, x, y, idx, seq_len, lens):
        self.x = x; self.y = y; self.idx = idx
        self.seq_len = seq_len; self.lens = lens
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        k = int(self.idx[i])
        L = int(self.lens[k])
        if L > self.seq_len:
            s = L - self.seq_len
            xs, ys = self.x[k][s:L], self.y[k][s:L]   # хвост контента (ответ ассистента)
        else:
            xs, ys = self.x[k][:self.seq_len], self.y[k][:self.seq_len]  # контент+паддинг (loss игнорит)
        xv = torch.from_numpy(np.ascontiguousarray(xs, dtype=np.int64).copy())
        yv = torch.from_numpy(np.ascontiguousarray(ys, dtype=np.int64).copy())
        return xv, yv


log("Pre-tokenization...")
X_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_x.npy")
Y_LOCAL = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_y.npy")
X_DRIVE = os.path.join(CKPT_DIR, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_x.npy")
Y_DRIVE = os.path.join(CKPT_DIR, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}_y.npy")
CACHE_LOCAL_NPZ = os.path.join(CONTENT, f"jarvis_mof_samples{'' if STAGE == 1 else '_stage2'}.npz")


def ensure_memmap_cache():
    if os.path.exists(X_LOCAL) and os.path.exists(Y_LOCAL):
        return np.load(X_LOCAL, mmap_mode="r"), np.load(Y_LOCAL, mmap_mode="r"), "memmap local"
    if os.path.exists(X_DRIVE) and os.path.exists(Y_DRIVE):
        try:
            shutil.copyfile(X_DRIVE, X_LOCAL)
            shutil.copyfile(Y_DRIVE, Y_LOCAL)
            return (np.load(X_LOCAL, mmap_mode="r"), np.load(Y_LOCAL, mmap_mode="r"),
                    "memmap c Диска")
        except Exception as e:
            log(f"  [WARN] копия memmap c Диска: {e}")
    for npz in (CACHE_LOCAL_NPZ, CACHE_DRIVE):
        if os.path.exists(npz):
            src = npz
            if npz is not CACHE_LOCAL_NPZ and "/content/drive" in npz:
                try:
                    log(f"  copy c Drive -> {CACHE_LOCAL_NPZ}")
                    shutil.copyfile(npz, CACHE_LOCAL_NPZ)
                    src = CACHE_LOCAL_NPZ
                except Exception as e:
                    log(f"  [WARN] копия npz с Диска: {e}")
            log(f"  извлекаю npz -> memmap: {src}")
            extract_npz_to_memmap(src, X_LOCAL, Y_LOCAL)
            return np.load(X_LOCAL, mmap_mode="r"), np.load(Y_LOCAL, mmap_mode="r"), "npz->memmap"
    return None


X = Y = None
src_kind = None
if os.path.exists(X_LOCAL) and os.path.exists(Y_LOCAL):
    X = np.load(X_LOCAL, mmap_mode="r")
    Y = np.load(Y_LOCAL, mmap_mode="r")
    log(f"  memmap cache: {X_LOCAL}")
else:
    got = ensure_memmap_cache()
    if got is None:
        tokenize_to_memmap(os.environ.get("FSTNET_DATA", "data/jarvis_full.json"),
                           cfg.max_seq_len, X_LOCAL, Y_LOCAL)
        X = np.load(X_LOCAL, mmap_mode="r")
        Y = np.load(Y_LOCAL, mmap_mode="r")
        for dst in (CACHE_LOCAL_NPZ, CACHE_DRIVE):
            try:
                save_npz_streaming(dst, (("x", X), ("y", Y)))
                log(f"  npz backup -> {dst}")
            except Exception as e:
                log(f"  [WARN] npz {dst} не записан: {e}")
    else:
        X, Y, src_kind = got
        log(f"  cache: {src_kind}")

rng_seed = int(os.environ.get("JARVIS_SEED", "42"))
rng = np.random.default_rng(rng_seed)
perm = rng.permutation(len(X))
val_n = max(int(len(X) * 0.03), 1)
val_idx = perm[:val_n]
tr_idx = perm[val_n:]
sample_n = int(os.environ.get("FSTNET_SAMPLES", "100000"))
if len(tr_idx) > sample_n:
    tr_idx = tr_idx[:sample_n]
    log(f"  subsample: {sample_n} (из {len(perm) - val_n})")
log(f"Train: {len(tr_idx)}, Val: {len(val_idx)}")

SEQ = int(os.environ.get("FSTNET_SEQ", "1024"))    # кроп: веса+грады=13.2GB,левые активации ~1.8GB; 2048² не влезал

log("  compting content lens...")
W = X.shape[1]
lens = np.empty(len(X), dtype=np.int64)
CH = 8192
for s in range(0, len(X), CH):
    b = np.asarray(X[s:s + CH]) != 0
    col = b[:, ::-1].argmax(axis=1)
    lens[s:s + CH] = np.where(b.any(axis=1), W - col, 0)
log(f"  lens: avg={lens.mean():.0f} p50={np.percentile(lens,50):.0f} "
    f"p95={np.percentile(lens,95):.0f} p99={np.percentile(lens,99):.0f} max={lens.max():.0f}")

SEQ = int(os.environ.get("FSTNET_SEQ", "512"))     # avg контента 347: 512 хватает, attention в 4x меньше чем 2048
train_ds = DS(X, Y, tr_idx, seq_len=SEQ, lens=lens)
val_ds = DS(X, Y, val_idx, seq_len=SEQ, lens=lens)
BATCH = int(os.environ.get("FSTNET_BATCH", "2"))    # T4 16GB: веса+грады 13.2GB; 2048² в attention не влезал
ACCUM = int(os.environ.get("FSTNET_ACCUM", "32"))    # эффективный батч 64
WORKERS = int(os.environ.get("FSTNET_WORKERS", "0"))
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
    pin_memory=False, persistent_workers=WORKERS > 0, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
    pin_memory=False, persistent_workers=WORKERS > 0)
VAL_MAX = int(os.environ.get("FSTNET_VAL_MAX", "512"))   # сэмпл валидации, а не все 15k (экономия часов)

EPOCHS = int(os.environ.get("FSTNET_EPOCHS", "1"))
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

COMPILE = os.environ.get("FSTNET_COMPILE", "").strip()
if cap < 8:
    # T4 (sm_7): torch.compile ~час компилирует первый батч и держит лишние
    # буферы -> OOM на backward (веса+грады 13.2GB из 14.56GB). Игнорируем
    # FSTNET_COMPILE всегда, чтобы залипший env в сессии не ломал запуск.
    if COMPILE not in ("", "0"):
        log(f"[WARN] FSTNET_COMPILE={COMPILE} игнорируется: на sm_{cap} "
            "torch.compile даёт OOM. Обучение без compile.")
    COMPILE = ""
if COMPILE not in ("", "0"):
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


step = step0
t0 = time.time()
best_val = float("inf")
mm.reset()  # пик VRAM считать только с тренировочного цикла (не с создания модели)
apply_phase(step / max(1, total_steps), freeze_w0=False)
model.train()

log(f"Training: {total_steps} steps | batch={BATCH} accum={ACCUM} (eff {BATCH*ACCUM}) | "
    f"lr={LEARN_RATE:.1e} | EPOCHS={EPOCHS} | seq={SEQ} | fields={cfg.n_fields} topk={cfg.gating_top_k}")

last_pulse = time.time()
PULSE = float(os.environ.get("FSTNET_PROGRESS_SEC", "15"))
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}",
                total=len(train_loader), unit="batch")
    opt.zero_grad(set_to_none=True)
    for it, (bx, by) in enumerate(pbar):
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
        mm.before_backward()
        loss.backward()
        if (it + 1) % ACCUM == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            sch.step()
            mm.after_step()
            step += 1
            pbar.set_postfix(opt=f"{step}/{total_steps}")

        now = time.time()
        if now - last_pulse >= PULSE:
            last_pulse = now
            el = now - t0
            eta = el / max(step - step0, 1) * (total_steps - step)
            log(f"  [pulse {it+1}/{len(train_loader)} батч] opt {step}/{total_steps} "
                f"({100*step/max(total_steps,1):.1f}%) | CE {ce.item():.4f} | "
                f"ETA {eta/60:.0f}min")

        if step % 25 == 0 and (it + 1) % ACCUM == 0:
            el = time.time() - t0
            eta = el / max(step - step0, 1) * (total_steps - step)
            curr_ratio = model.blocks[0].ffn.W0g.binarize
            log(f"Step {step}/{total_steps} | CE {ce.item():.4f}"
                f"{' ORTH ' + f'{orth.item():.4f}' if use_orth else ''} | "
                f"β {curr_ratio:.2f} | LR {sch.get_last_lr()[0]:.2e} | "
                f"{mm.report()} | ETA {eta/60:.0f}min")

        if it > 0 and it % 500 == 0:
            state = {"step": step, "model_state": model.state_dict(), "config": cfg}
            torch.save(state, CKPT_LOCAL)
            uploader.submit(CKPT_LOCAL, CKPT_DRIVE)
            log(f"  >> autosave (batch {it})")
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
                torch.save(state, CKPT_LOCAL)
                uploader.submit(CKPT_LOCAL, CKPT_DRIVE)
                log(f"  >> best saved (val {best_val:.4f})")
            model.train()

state = {"step": step, "model_state": model.state_dict(), "config": cfg}
torch.save(state, FINAl_LOCAL)
uploader.submit(FINAl_LOCAL, os.path.join(CKPT_DIR, "final.pt"))
log(f"DONE. Step={step}, best_val={best_val:.4f}")
log(f"Следующее: 1-bit export (S3) + bitnet.cpp fork (см. SPEC_3B_MOF.md).")