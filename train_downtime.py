#!/usr/bin/env python3
"""Continue training FST-Net 64M on more data."""
import os, json, time, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from config import FSTConfig
from model.core import FSTNetCore
from tokenizers import Tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100
SEQ_LEN = 512

def make_dataset(path, tok, seq_len):
    """Convert ChatML data to training samples."""
    data = json.load(open(path))
    samples = []
    for conv in data:
        ids = []
        for role, content in conv:
            ids += tok.encode(f"{IM_START}{role}\n{content}{IM_END}").ids
            ids += tok.encode("\n").ids
        if len(ids) < 8 or len(ids) > seq_len:
            continue
        # Find assistant start
        asst_ids = tok.encode(f"{IM_START}assistant\n").ids
        # Find position of assistant turn
        text_ids = ids[:]
        asst_pos = len(text_ids) // 2  # fallback
        for i in range(len(text_ids) - len(asst_ids) + 1):
            if text_ids[i:i+len(asst_ids)] == asst_ids:
                asst_pos = i
                break
        
        x, y = [], []
        for i in range(seq_len):
            j = i + 1
            x.append(ids[i] if i < len(ids) else PAD)
            y.append(ids[j] if (j >= asst_pos and j < len(ids)) else IGNORE)
        samples.append((x, y))
    return samples

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train(
    pretrained="checkpoints/ft1024/final.pt",
    data_path="data/train_extra.json",
    save_dir="checkpoints/downtime",
    epochs=3,
    lr=2e-5,
    batch_size=2,
    accum=8,
):
    device = "cpu"
    config = FSTConfig(vocab_size=32770, d_model=896, n_heads=16, n_kv_heads=4, d_ff=3584, n_layers=3, max_seq_len=SEQ_LEN)
    model = FSTNetCore(config)
    params, _ = model.count_parameters()
    print(f"Params: {params/1e6:.1f}M", flush=True)

    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state"].items() if "causal_mask" not in k}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded {pretrained}", flush=True)

    tok = Tokenizer.from_file("tokenizer/fst_bpe.json")
    tok.no_truncation(); tok.no_padding()

    print("Preparing dataset...", flush=True)
    samples = make_dataset(data_path, tok, SEQ_LEN)
    print(f"Samples: {len(samples)}", flush=True)

    ds = SimpleDS(samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, foreach=False)
    total_steps = epochs * len(ds) // (batch_size * accum)
    warmup = total_steps // 10

    def lr_fn(s):
        if s < warmup: return (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

    model = model.to(device)
    model.train()

    step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(accum):
                try:
                    bx_m, by_m = next(iter(loader))
                except StopIteration:
                    bx_m, by_m = bx, by
                bx_m, by_m = bx_m.to(device), by_m.to(device)

                h, _ = model(bx_m, target_cycles=4, return_hidden=True)
                loss_sum = torch.tensor(0.0, device=device)
                n_valid = 0
                for s in range(0, SEQ_LEN, 64):
                    e = min(s + 64, SEQ_LEN)
                    logits = model.head(h[:, s:e])
                    loss = criterion(logits.view(-1, config.vocab_size), by_m[:, s:e].reshape(-1))
                    loss_sum += loss
                    n_valid += (by_m[:, s:e] != IGNORE).sum().item()
                loss = loss_sum / max(n_valid, 1) / accum
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            pbar.set_postfix(loss=f"{accum_loss/accum:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if step % 50 == 0:
                pbar.write(f"Step {step}/{total_steps} | Loss: {accum_loss/accum:.4f} | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    os.makedirs(save_dir, exist_ok=True)
    final = os.path.join(save_dir, "final.pt")
    torch.save({"step": step, "model_state": model.state_dict(), "config": config}, final)
    print(f"Done: {step} steps", flush=True)
    print(f"Final: {final}", flush=True)

if __name__ == "__main__":
    train()
