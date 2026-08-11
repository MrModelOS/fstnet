#!/usr/bin/env python3
"""FST-Net training for Google Colab T4 GPU. Saves logs + checkpoints."""
import os, json, time, subprocess
from datetime import datetime

subprocess.run(["pip", "install", "-q", "transformers", "datasets", "tokenizers", "tqdm"], check=True)

if not os.path.exists("config.py"):
    print("ERROR: run from fstnet repo folder")
    exit(1)

# ── Setup logging ───────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints/downtime", exist_ok=True)
log_file = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log(f"GPU: {os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip()}")
log(f"PyTorch: {torch.__version__}" if 'torch' in dir() else "")

# ── Data ────────────────────────────────────────────────
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

# ── Training ────────────────────────────────────────────
from config import FSTConfig
from model.core import FSTNetCore
from tokenizers import Tokenizer
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100
SEQ = 512

def make_samples(path, tok):
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

class DS(torch.utils.data.Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        return torch.tensor(self.s[i][0], dtype=torch.long), torch.tensor(self.s[i][1], dtype=torch.long)

cfg = FSTConfig(vocab_size=32770, d_model=896, n_heads=16, n_kv_heads=4, d_ff=3584, n_layers=3, max_seq_len=SEQ)
model = FSTNetCore(cfg)
ckpt = torch.load("checkpoints/ft1024/final.pt", map_location="cpu", weights_only=False)
sd = {k:v for k,v in ckpt["model_state"].items() if "causal_mask" not in k}
model.load_state_dict(sd, strict=False)
log(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

tok = Tokenizer.from_file("tokenizer/fst_bpe.json")
samples = make_samples("data/train_extra.json", tok)
log(f"Samples: {len(samples)}")
ds = DS(samples)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

opt = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
total_steps = 3 * len(ds) // 4
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5*(1+__import__('math').cos(math.pi*s/total_steps)))
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

device = "cuda"
model = model.to(device)
model.train()
step = 0
start_time = time.time()

log(f"Starting training: {total_steps} steps, batch=4, seq={SEQ}")
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
        
        loss_val = ls.item()/max(nv,1)
        if step % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / step * (total_steps - step)
            log(f"Step {step}/{total_steps} | Loss: {loss_val:.4f} | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB | ETA: {eta/60:.0f}min")

# ── Save ────────────────────────────────────────────────
torch.save({"step": step, "model_state": model.state_dict(), "config": cfg}, "checkpoints/downtime/final.pt")
elapsed = time.time() - start_time
log("="*60)
log(f"DONE: {step} steps in {elapsed/60:.1f} min")
log(f"Checkpoint: checkpoints/downtime/final.pt")
log(f"Log: {log_file}")
