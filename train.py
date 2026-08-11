import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import FSTConfig
from model.core import FSTNetCore
from data import LocalTextDataset, SyntheticTextDataset, InstructDataset, IGNORE


def build_dataset(data_dir: str, use_synthetic: bool, use_instruct: bool, max_samples: int, seq_len: int, tokenizer_path: str = "tokenizer/fst_bpe.json"):
    if use_instruct:
        print("Using InstructDataset", flush=True)
        return InstructDataset(data_dir=data_dir, seq_len=seq_len, max_samples=max_samples)
    if use_synthetic:
        print("Using SyntheticTextDataset", flush=True)
        return SyntheticTextDataset(tokenizer_path=tokenizer_path, seq_len=seq_len, max_samples=max_samples)
    try:
        return LocalTextDataset(data_dir=data_dir, tokenizer_path=tokenizer_path, seq_len=seq_len, max_samples=max_samples)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Local data failed: {e}", flush=True)
        print("Falling back to SyntheticTextDataset", flush=True)
        return SyntheticTextDataset(tokenizer_path=tokenizer_path, seq_len=seq_len, max_samples=max_samples)


def train(
    data_dir: str = "data/local",
    use_synthetic: bool = False,
    use_instruct: bool = False,
    max_samples: int = 50_000,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    learning_rate: float = 4e-4,
    train_cycles: int = 4,
    epochs: int = 1,
    save_dir: str = "checkpoints",
    log_every: int = 10,
    save_every: int = 1000,
    pretrained: str = None,
):
    effective_batch = batch_size * grad_accum_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Batch: {batch_size} x {grad_accum_steps} accum = effective {effective_batch}", flush=True)

    config = FSTConfig()
    model = FSTNetCore(config).to(device)
    params, vram_mb = model.count_parameters()
    print(f"Params: {params:,} | Weights FP32: {params * 4 / 1024**2:.1f} MB", flush=True)

    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        old_vocab, new_vocab = model.load_checkpoint_into(ckpt["model_state"])
        print(f"Loaded pretrained {pretrained} (step {ckpt.get('step', '?')})", flush=True)
        if old_vocab != new_vocab:
            print(f"  Vocab expanded: {old_vocab} -> {new_vocab} (new special tokens)", flush=True)
    model = model.to(device)

    use_fp16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    if use_fp16:
        model = model.half()
        print("AMP FP16 enabled (tensor cores)", flush=True)
    else:
        print("FP32 mode (stable, MX450 has no tensor cores)", flush=True)

    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...", flush=True)
    dataset = build_dataset(data_dir, use_synthetic, use_instruct, max_samples, config.max_seq_len, config.tokenizer_path)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty!")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        foreach=False,
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
    model.train()

    start_time = time.time()
    step = 0
    resume_path = os.path.join(save_dir, "fst_resume.pt")
    if os.path.exists(resume_path) and pretrained is None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ckpt["step"]
        for _ in range(step):
            scheduler.step()
        print(f"Resumed from step {step}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  FST-Net Training", flush=True)
    print(f"  Dataset:   {len(dataset):,} samples", flush=True)
    print(f"  Eff batch: {effective_batch}", flush=True)
    print(f"  Steps:     {total_steps}", flush=True)
    print(f"  LR:        {learning_rate}", flush=True)
    print(f"  Cycles:    {train_cycles}", flush=True)
    print(f"{'='*60}", flush=True)

    pbar = tqdm(
        total=total_steps,
        desc="Training",
        unit="step",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    pbar.update(step)

    data_iter = iter(loader)

    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            try:
                batch_x, batch_y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch_x, batch_y = next(data_iter)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward: скрытое состояние вместо полных logits (экономия VRAM)
            h, cycles = model(batch_x, target_cycles=train_cycles, return_hidden=True)

            # Chunked loss: logits считаем частями по seq-оси (KILLS OOM)
            # [B, T, D] -> [B, chunk, D] -> head -> [B, chunk, vocab]
            chunk_size = 32
            B, T, _ = h.shape
            loss_sum = torch.tensor(0.0, device=device)
            n_valid = 0

            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                logits_chunk = model.head(h[:, s:e])
                loss_chunk = criterion(
                    logits_chunk.view(-1, config.vocab_size),
                    batch_y[:, s:e].reshape(-1),
                )
                loss_sum = loss_sum + loss_chunk
                n_valid += (batch_y[:, s:e] != IGNORE).sum().item()

            loss = loss_sum / max(n_valid, 1) / grad_accum_steps
            loss.backward()

            accum_loss += loss.item() * grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1

        avg_loss = accum_loss / grad_accum_steps
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            tqdm.write(f"WARN: NaN loss at step {step}, resuming from checkpoint...")
            ckpt_path = os.path.join(save_dir, "fst_resume.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                tqdm.write("  Restored weights from fst_resume.pt")
            continue

        torch.save(
            {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            },
            os.path.join(save_dir, "fst_resume.pt"),
        )

        avg_loss = accum_loss / grad_accum_steps
        ppl = math.exp(min(avg_loss, 20))
        cur_lr = scheduler.get_last_lr()[0]

        mem_info = ""
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            mem_info = f" | VRAM {allocated:.0f}MB"

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            ppl=f"{ppl:.1f}",
            lr=f"{cur_lr:.2e}",
            cycles=f"{cycles}",
            refresh=True,
        )
        pbar.update(1)
        pbar.refresh()

        if step % log_every == 0:
            elapsed = time.time() - start_time
            samples_processed = step * effective_batch
            samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
            tqdm.write(
                f"Step {step:>6d}/{total_steps} | Loss: {avg_loss:.4f} | "
                f"PPL: {ppl:.1f} | LR: {cur_lr:.2e} | {samples_per_sec:.0f} samp/s{mem_info}",
                end="",
            )
            tqdm.write("")

        if step % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"fst_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                ckpt_path,
            )
            tqdm.write(f"  >> Checkpoint: {ckpt_path}")

    pbar.close()

    final_path = os.path.join(save_dir, "fst_final.pt")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "config": config,
            "total_samples": step * effective_batch,
        },
        final_path,
    )

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"  Training Complete", flush=True)
    print(f"  Steps: {step} | Samples: {step * effective_batch:,}", flush=True)
    print(f"  Time: {total_time:.1f} min", flush=True)
    print(f"  Final: {final_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FST-Net Training")
    parser.add_argument("--data-dir", default="data/local")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None, help="Pretrained checkpoint to fine-tune from")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        use_instruct=args.instruct,
        max_samples=args.samples,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        learning_rate=args.lr,
        train_cycles=args.cycles,
        epochs=args.epochs,
        save_dir=args.save_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        pretrained=args.ckpt,
    )
