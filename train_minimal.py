import os, json, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from config import FSTConfig
from model.core import FSTNetCore
from tokenizers import Tokenizer

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100
SEQ = 256

def make_ds(path, tok):
    data = json.load(open(path))
    samples = []
    for conv in data[:5000]:
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
        x, y = self.s[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

device = "cuda"
cfg = FSTConfig(vocab_size=32770, d_model=896, n_heads=16, n_kv_heads=4, d_ff=3584, n_layers=3, max_seq_len=SEQ)
model = FSTNetCore(cfg)
ckpt = torch.load("checkpoints/ft1024/final.pt", map_location="cpu", weights_only=False)
sd = {k: v for k, v in ckpt["model_state"].items() if "causal_mask" not in k}
model.load_state_dict(sd, strict=False)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

tok = Tokenizer.from_file("tokenizer/fst_bpe.json")
samples = make_ds("data/train_extra.json", tok)
print(f"Samples: {len(samples)}", flush=True)
ds = DS(samples)
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

opt = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
steps = len(ds)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5*(1+math.cos(math.pi*s/steps)))
crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

model = model.to(device)
model.train()
step = 0
for epoch in range(3):
    pbar = tqdm(loader, desc=f"E{epoch+1}")
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
        if step % 20 == 0:
            pbar.set_postfix(loss=f"{ls.item()/max(nv,1):.4f}", vram=f"{torch.cuda.memory_allocated()/1024**2:.0f}MB")

os.makedirs("checkpoints/downtime", exist_ok=True)
torch.save({"step": step, "model_state": model.state_dict(), "config": cfg}, "checkpoints/downtime/final.pt")
print(f"Done: {step} steps", flush=True)
