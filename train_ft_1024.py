#!/usr/bin/env python3
import os, math, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from config import FSTConfig
from model.core import FSTNetCore
from data import InstructDataset, IGNORE

def make_config():
    return FSTConfig(vocab_size=32770, d_model=896, n_heads=16, n_kv_heads=4,
                     d_ff=3584, n_layers=3, max_seq_len=512, max_cycles=4)

def train(pretrained="checkpoints/100m/final.pt", save_dir="checkpoints/ft1024"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_config()
    model = FSTNetCore(config)
    params, _ = model.count_parameters()
    print(f"Params: {params/1e6:.1f}M", flush=True)

    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state"].items() if "causal_mask" not in k}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded {pretrained}", flush=True)

    os.makedirs(save_dir, exist_ok=True)
    ds1 = InstructDataset("data/mix_512", seq_len=512, max_samples=100000)
    ds2 = InstructDataset("data/fc_512", seq_len=512, max_samples=100000)
    ds = ConcatDataset([ds1, ds2])
    print(f"Dataset: {len(ds)} samples", flush=True)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    total_steps = 800
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05, foreach=False)
    warmup = 80

    def lr_fn(s):
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    model = model.to(device)
    model.train()

    start = time.time()
    step = 0
    pbar = tqdm(total=total_steps, desc="FT1024", unit="step")
    data_iter = iter(loader)

    while step < total_steps:
        optimizer.zero_grad()
        accum = 0.0
        for _ in range(4):
            try:
                bx, by = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                bx, by = next(data_iter)
            bx, by = bx.to(device), by.to(device)

            h, _ = model(bx, target_cycles=4, return_hidden=True)
            B, T, _ = h.shape
            loss_sum = torch.tensor(0.0, device=device)
            n_valid = 0
            for s in range(0, T, 64):
                e = min(s+64, T)
                logits = model.head(h[:,s:e])
                loss = nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    by[:,s:e].reshape(-1),
                    ignore_index=IGNORE,
                    reduction="sum",
                )
                loss_sum += loss
                n_valid += (by[:,s:e] != IGNORE).sum().item()
            loss = loss_sum / max(n_valid, 1) / 4
            loss.backward()
            accum += loss.item(); torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        avg = accum / 16
        torch.save({"step": step, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "config": config},
                    os.path.join(save_dir, "resume.pt"))
        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        pbar.update(1)
        if step % 10 == 0:
            pbar.write(f"Step {step}/{total_steps} | Loss {avg:.4f} | PPL {math.exp(min(avg,20)):.1f} | VRAM {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    pbar.close()
    final = os.path.join(save_dir, "final.pt")
    torch.save({"step": step, "model_state": model.state_dict(), "config": config}, final)
    print(f"Done: {step} steps, {(time.time()-start)/60:.0f} min", flush=True)

if __name__ == "__main__":
    train()
