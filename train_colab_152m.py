#!/usr/bin/env python3
"""FST-Net 150M training. Optimized: fp16 + cosine LR + validation."""
import os, json, time, subprocess, math, random
from datetime import datetime

subprocess.run(["pip", "install", "-q", "transformers", "datasets", "tokenizers", "tqdm"], check=True)

from colab_drive import setup_checkpoint_dir
CKPT_DIR = setup_checkpoint_dir(subdir="152m")

os.makedirs("logs", exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
log_file = f"logs/train_152m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log(f"GPU: {os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip()}")

from datasets import load_dataset
os.makedirs("data", exist_ok=True)
all_convs = []

log("Loading datasets...")
# CodeAlpaca
try:
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    for d in ds:
        instr, inp, out = d.get("instruction","").strip(), d.get("input","").strip(), d.get("output","").strip()
        if instr and out:
            all_convs.append([("user", f"{instr}\n{inp}" if inp else instr), ("assistant", out)])
    log(f"  codealpaca: {len(ds)}")
except Exception as e: log(f"  codealpaca skip: {e}")

# SlimOrca
try:
    ds = load_dataset("Open-Orca/SlimOrca-Dedup", split="train", streaming=True)
    cnt = 0
    for d in ds:
        msgs = d.get("conversations", [])
        conv = [(m["from"], m["value"]) for m in msgs if m["from"] in ("human","gpt")]
        if len(conv) >= 2: all_convs.append(conv); cnt += 1
        if cnt >= 10000: break
    log(f"  slimorca: {cnt}")
except Exception as e: log(f"  slimorca skip: {e}")

# UltraChat
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
cnt = 0
for d in ds:
    msgs = [(m["role"], m["content"]) for m in d.get("messages", []) if m["role"] in ("user","assistant")]
    if len(msgs) >= 2: all_convs.append(msgs); cnt += 1
    if cnt >= 10000: break
log(f"  ultrachat: {cnt}")

# GSM8K
ds = load_dataset("openai/gsm8k", "main", split="train")
for d in ds: all_convs.append([("user", d["question"]), ("assistant", d["answer"])])
log(f"  gsm8k: {len(ds)}")

with open("data/train_full.json", "w") as f: json.dump(all_convs, f)
log(f"Total: {len(all_convs)}")

# Model
import sys, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from model.core import FSTNetCore
from config_152m import FSTConfig152M

cfg = FSTConfig152M()
model = FSTNetCore(cfg)
params = sum(p.numel() for p in model.parameters())
log(f"Params: {params/1e6:.1f}M")

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/fst_bpe.json")
IM_S, IM_E = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100
SEQ = cfg.max_seq_len

def make_samples(path):
    data = json.load(open(path))
    samples = []
    for conv in data:
        ids = []
        for role, content in conv: ids += tok.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
        if len(ids) < 8 or len(ids) > SEQ: continue
        asst = tok.encode(f"{IM_S}assistant\n").ids
        ap = len(ids) // 2
        for i in range(len(ids) - len(asst) + 1):
            if ids[i:i+len(asst)] == asst: ap = i; break
        x, y = [], []
        for i in range(SEQ):
            j = i + 1
            x.append(ids[i] if i < len(ids) else PAD)
            y.append(ids[j] if (j >= ap and j < len(ids)) else IGNORE)
        samples.append((x, y))
    return samples

class DS(torch.utils.data.Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        return torch.tensor(self.s[i][0], dtype=torch.long), torch.tensor(self.s[i][1], dtype=torch.long)

all_samples = make_samples("data/train_full.json")
random.seed(42); random.shuffle(all_samples)
val_split = int(len(all_samples) * 0.05)
val_samples = all_samples[:val_split]
train_samples = all_samples[val_split:]
log(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

train_ds = DS(train_samples)
val_ds = DS(val_samples)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Optimizer: lr=4e-4 with cosine + warmup
ACCUM = 1
BATCH = 32
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, foreach=False, fused=True)
total_steps = 3 * len(train_ds) // (BATCH * ACCUM)
warmup = total_steps // 10

def lr_fn(s):
    if s < warmup: return (s + 1) / warmup
    p = (s - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))

sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

device = "cuda"
model = model.to(device)

# bfloat16 на Ampere+, иначе fp16 (T4 не поддерживает bf16 в тензорных ядрах)
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
log(f"Compute dtype: {COMPUTE_DTYPE} (bf16={USE_BF16})")
scaler = None if USE_BF16 else GradScaler()

try:
    model = torch.compile(model)
    log("model = torch.compile() OK")
except Exception as e:
    log(f"torch.compile skip: {e}")

model.train()
step = 0
start_time = time.time()
best_val = float("inf")

log(f"Training: {total_steps} steps, batch={BATCH}, accum={ACCUM}, lr=4e-4, dtype={COMPUTE_DTYPE}")
log("="*60)

for epoch in range(3):
    pbar = tqdm(train_loader, desc=f"E{epoch+1}/3")
    opt.zero_grad(set_to_none=True)
    for bx, by in pbar:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            h, _ = model(bx, target_cycles=4, return_hidden=True)
            ls = torch.tensor(0.0, device=device)
            nv = 0
            for s in range(0, SEQ, 64):
                e = min(s+64, SEQ)
                l = crit(model.head(h[:,s:e]).view(-1, cfg.vocab_size), by[:,s:e].reshape(-1))
                ls += l
                nv += (by[:,s:e] != IGNORE).sum().item()
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
            elapsed = time.time() - start_time
            eta = elapsed/step*(total_steps-step)
            log(f"Step {step}/{total_steps} | Loss: {ls.item()/max(nv,1):.4f} | LR: {sch.get_last_lr()[0]:.2e} | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB | ETA: {eta/60:.0f}min")
        
        # Validation every 500 steps
        if step % 500 == 0:
            model.eval()
            val_loss, val_n = 0.0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                        h, _ = model(vx, target_cycles=4, return_hidden=True)
                        for s in range(0, SEQ, 64):
                            e = min(s+64, SEQ)
                            l = crit(model.head(h[:,s:e]).view(-1, cfg.vocab_size), vy[:,s:e].reshape(-1))
                            val_loss += l.item()
                            val_n += (vy[:,s:e] != IGNORE).sum().item()
            val_avg = val_loss / max(val_n, 1)
            log(f"  VAL LOSS: {val_avg:.4f}")
            if val_avg < best_val:
                best_val = val_avg
                torch.save({"step": step, "model_state": model.state_dict(), "config": cfg}, os.path.join(CKPT_DIR, "best.pt"))
                log(f"  >> Best checkpoint saved!")
            model.train()

torch.save({"step": step, "model_state": model.state_dict(), "config": cfg}, os.path.join(CKPT_DIR, "final.pt"))
elapsed = time.time() - start_time
log("="*60)
log(f"DONE: {step} steps in {elapsed/60:.1f} min | Best val: {best_val:.4f}")
