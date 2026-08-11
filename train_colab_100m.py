#!/usr/bin/env python3
"""FST-Net 100M training for Google Colab T4 GPU."""
import os, json, time, subprocess
from datetime import datetime

subprocess.run(["pip", "install", "-q", "transformers", "datasets", "tokenizers", "tqdm"], check=True)

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints/100m", exist_ok=True)
log_file = f"logs/train_100m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log(f"GPU: {os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip()}")

# Data
from datasets import load_dataset
os.makedirs("data", exist_ok=True)

log("Loading ultrachat_200k...")
ds1 = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
ultrachat = []
for i, d in enumerate(ds1):
    msgs = [(m["role"], m["content"]) for m in d.get("messages", []) if m["role"] in ("user","assistant")]
    if len(msgs) >= 2:
        ultrachat.append(msgs)
    if i >= 10000:
        break
log(f"  ultrachat: {len(ultrachat)}")

log("Loading gsm8k...")
ds2 = load_dataset("openai/gsm8k", "main", split="train")
gsm8k = [[("user", d["question"]), ("assistant", d["answer"])] for d in ds2]
log(f"  gsm8k: {len(gsm8k)}")

with open("data/train_extra.json", "w") as f:
    json.dump(ultrachat + gsm8k, f)
log(f"Total: {len(ultrachat) + len(gsm8k)} samples")

# Model
import sys
sys.path.insert(0, os.getcwd())
from model.core import FSTNetCore
from config_100m import FSTConfig100M

cfg = FSTConfig100M()
model = FSTNetCore(cfg)
params = sum(p.numel() for p in model.parameters())
log(f"Params: {params/1e6:.1f}M")

# Tokenizer
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
        for role, content in conv:
            ids += tok.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
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

import torch, torch.nn as nn
class DS(torch.utils.data.Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        return torch.tensor(self.s[i][0], dtype=torch.long), torch.tensor(self.s[i][1], dtype=torch.long)

from torch.utils.data import DataLoader
from tqdm import tqdm

samples = make_samples("data/train_extra.json")
log(f"Samples: {len(samples)}")
ds = DS(samples)
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
total_steps = 3 * len(ds) // 2
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5*(1+__import__('math').cos(math.pi*s/total_steps)))
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

device = "cuda"
model = model.to(device)
model.train()
step = 0
start_time = time.time()

log(f"Training: {total_steps} steps, batch=2, seq={SEQ}")
log("="*60)

for epoch in range(3):
    pbar = tqdm(loader, desc=f"E{epoch+1}/3")
    for bx, by in pbar:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        h, _ = model(bx, target_cycles=4, return_hidden=True)
        ls = torch.tensor(0.0, device=device)
        nv = 0
        for s in range(0, SEQ, 64):
            e = min(s+64, SEQ)
            l = crit(model.head(h[:,s:e]).view(-1, cfg.vocab_size), by[:,s:e].reshape(-1))
            ls += l
            nv += (by[:,s:e] != IGNORE).sum().item()
        (ls/max(nv,1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        step += 1
        if step % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / step * (total_steps - step)
            log(f"Step {step}/{total_steps} | Loss: {ls.item()/max(nv,1):.4f} | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB | ETA: {eta/60:.0f}min")

torch.save({"step": step, "model_state": model.state_dict(), "config": cfg}, "checkpoints/100m/final.pt")
elapsed = time.time() - start_time
log("="*60)
log(f"DONE: {step} steps in {elapsed/60:.1f} min")
