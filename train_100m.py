"""Запуск обучения большой модели FST-Net 100M (~98M параметров).

Конфиг: d_model=1024, n_heads=16, n_kv_heads=4, d_ff=4096, n_layers=4
VRAM: ~1.6GB (weights + grads + optimizer) — влезает в MX450 2GB.

Запуск:
  nohup python3 train_100m.py > /tmp/train_100m.log 2>&1 &
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import FSTConfig
from model.core import FSTNetCore
from data import InstructDataset, IGNORE


def make_config():
    # 64M — влезает в MX450 с gradient checkpointing
    # d_model=896, n_heads=16, n_kv_heads=4, d_ff=3584, n_layers=3
    return FSTConfig(
        vocab_size=32770,
        d_model=896,
        n_heads=16,
        n_kv_heads=4,
        d_ff=3584,
        n_layers=3,
        max_cycles=4,
        max_seq_len=256,
    )


def train(
    data_dir: str = "data/mix",
    batch_size: int = 1,
    grad_accum_steps: int = 16,
    learning_rate: float = 8e-5,
    train_cycles: int = 4,
    epochs: int = 1,
    save_dir: str = "checkpoints/100m",
    log_every: int = 20,
    save_every: int = 500,
    max_samples: int = 24000,
):
    effective_batch = batch_size * grad_accum_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Batch: {batch_size} x {grad_accum_steps} accum = effective {effective_batch}", flush=True)

    config = make_config()
    model = FSTNetCore(config)  # создаём на CPU
    params, _ = model.count_parameters()
    print(f"Params: {params:,} | FP32 weights: {params*4/1024**2:.0f} MB", flush=True)

    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...", flush=True)
    dataset = InstructDataset(data_dir=data_dir, seq_len=config.max_seq_len, max_samples=max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty!")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
        weight_decay=0.1, foreach=False,
    )

    total_steps = epochs * (len(dataset) // effective_batch)
    warmup = max(50, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

    # resume — грузим на CPU потом .to(device)
    start_time = time.time()
    step = 0
    resume_path = os.path.join(save_dir, "resume.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ckpt["step"]
        for _ in range(step):
            scheduler.step()
        print(f"Resumed from step {step}", flush=True)

    model = model.to(device)
    # перенести optimizer state на GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.train()

    print(f"\n{'='*60}", flush=True)
    print(f"  FST-Net 100M Training", flush=True)
    print(f"  Params: {params:,}", flush=True)
    print(f"  Dataset: {len(dataset):,} samples", flush=True)
    print(f"  Steps: {total_steps}", flush=True)
    print(f"  LR: {learning_rate}", flush=True)
    print(f"{'='*60}", flush=True)

    pbar = tqdm(total=total_steps, desc="Training 64M", unit="step", ncols=100)
    pbar.update(step)
    data_iter = iter(loader)

    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            try:
                bx, by = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                bx, by = next(data_iter)
            bx, by = bx.to(device), by.to(device)

            h, _ = model(bx, target_cycles=train_cycles, return_hidden=True)

            # chunked loss
            chunk_size = 32
            B, T, _ = h.shape
            loss_sum = torch.tensor(0.0, device=device)
            n_valid = 0
            n_chunks = (T + chunk_size - 1) // chunk_size

            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                logits_chunk = model.head(h[:, s:e])
                loss_chunk = criterion(
                    logits_chunk.view(-1, config.vocab_size),
                    by[:, s:e].reshape(-1),
                )
                loss_sum += loss_chunk
                n_valid += (by[:, s:e] != IGNORE).sum().item()

            loss = loss_sum / max(n_valid, 1) / grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        avg_loss = accum_loss / grad_accum_steps
        if math.isnan(avg_loss):
            tqdm.write(f"WARN NaN at step {step}")
            continue

        torch.save({"step": step, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "config": config},
                    os.path.join(save_dir, "resume.pt"))

        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        pbar.update(1)

        if step % log_every == 0:
            pbar.write(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | "
                        f"PPL: {math.exp(min(avg_loss,20)):.1f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}")
            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**2
                pbar.write(f"  VRAM: {mem:.0f} MB")

        if step % save_every == 0:
            path = os.path.join(save_dir, f"step_{step}.pt")
            torch.save({"step": step, "model_state": model.state_dict(),
                        "config": config, "loss": avg_loss}, path)
            pbar.write(f"  >> Checkpoint: {path}")

    pbar.close()
    final_path = os.path.join(save_dir, "final.pt")
    torch.save({"step": step, "model_state": model.state_dict(), "config": config},
                final_path)
    print(f"\nDone: {step} steps, {step*effective_batch:,} samples, "
          f"{(time.time()-start_time)/60:.1f} min", flush=True)
    print(f"Final: {final_path}", flush=True)


if __name__ == "__main__":
    train()
