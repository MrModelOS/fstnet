"""Премиум-смесь для 64M FST-Net.

Пропорции:
  35% — диалог (ultrachat + UltraFeedback)
  40% — код (python_code_instructions + the-stack filtered)
  15% — CoT/логика (gsm8k)
  10% — существующий mix (CodeAlpaca + SmolTalk)

Формат: ChatML, loss только на assistant-части.
"""
import os, json, random, sys
from datasets import load_dataset
from tokenizers import Tokenizer

TOK = "tokenizer/fst_bpe.json"
OUT_DIR = "data/premium"
SEQ_LEN = 256
MAX_TOTAL = 40_000  # ~2000 steps с eff batch 20

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100

os.makedirs(OUT_DIR, exist_ok=True)
tok = Tokenizer.from_file(TOK)
tok.no_truncation(); tok.no_padding()

def to_chatml_ids(messages):
    """messages: list of (role, content) -> ids"""
    ids = []
    for role, content in messages:
        ids += tok.encode(f"{IM_START}{role}\n{content}{IM_END}").ids
        ids += tok.encode("\n").ids
    return ids

def find_assistant_start(ids):
    im = tok.token_to_id(IM_START)
    asst_tok = tok.encode("assistant").ids[0]
    starts = [i for i, t in enumerate(ids) if t == im]
    for s in starts:
        if s + 1 < len(ids) and ids[s + 1] == asst_tok:
            return s
    return len(ids) // 2

def to_sample(ids):
    asst = find_assistant_start(ids)
    x, y = [], []
    for i in range(SEQ_LEN):
        j = i + 1
        x.append(ids[i] if i < len(ids) else PAD)
        y.append(ids[j] if (j >= asst and j < len(ids)) else IGNORE)
    return x, y

def add_samples(samples, label, max_n):
    added = 0
    for s in samples:
        if added >= max_n:
            break
        try:
            ids = to_chatml_ids(s)
        except Exception:
            continue
        if len(ids) < 8 or len(ids) > SEQ_LEN:
            continue
        x, y = to_sample(ids)
        all_x.append(x); all_y.append(y); added += 1
    print(f"  {label}: {added}", flush=True)
    return added

all_x, all_y = []
# ── 1. Диалог (35%) ──
print("=== Dialog ===", flush=True)

# ultrachat_200k
try:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    samples = []
    for d in ds:
        msgs = [(m["role"], m["content"]) for m in d.get("messages", []) if m["role"] in ("user","assistant")]
        if len(msgs) >= 2:
            samples.append(msgs)
        if len(samples) >= 8000:
            break
    add_samples(samples, "ultrachat_200k", 8000)
except Exception as e:
    print(f"  ultrachat skip: {e}", flush=True)

# UltraFeedback
try:
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    samples = []
    for d in ds:
        msgs = [("user", d.get("instruction","")), ("assistant", d.get("output",""))]
        if msgs[0][1] and msgs[1][1]:
            samples.append(msgs)
        if len(samples) >= 6000:
            break
    add_samples(samples, "UltraFeedback", 6000)
except Exception as e:
    print(f"  UltraFeedback skip: {e}", flush=True)

# ── 2. Код (40%) ──
print("=== Code ===", flush=True)

# python_code_instructions_18k_alpaca
try:
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    samples = []
    for d in ds:
        instr = d.get("instruction","").strip()
        inp = d.get("input","").strip()
        out = d.get("output","").strip()
        if not out:
            continue
        prompt = f"{instr}\n{inp}" if inp else instr
        samples.append([("user", prompt), ("assistant", out)])
        if len(samples) >= 10000:
            break
    add_samples(samples, "python_code_18k", 10000)
except Exception as e:
    print(f"  python_code skip: {e}", flush=True)

# the-stack-dedup filtered (Python, C, Bash, Rust)
try:
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train", streaming=True, trust_remote_code=True)
    samples = []
    for d in ds:
        content = d.get("content","").strip()
        if not content or len(content) > 3000 or len(content) < 100:
            continue
        lines = content.split("\n")
        # first 3 lines as signature, rest as body
        sig = "\n".join(lines[:3])
        body = "\n".join(lines[:min(20, len(lines))])
        samples.append([("user", f"Complete this Python function:\n{sig}"), ("assistant", body)])
        if len(samples) >= 5000:
            break
    add_samples(samples, "the-stack-python", 5000)
except Exception as e:
    print(f"  the-stack skip: {e}", flush=True)

# ── 3. CoT / логика (15%) ──
print("=== CoT ===", flush=True)
try:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    samples = []
    for d in ds:
        samples.append([("user", d["question"]), ("assistant", d["answer"])])
    add_samples(samples, "gsm8k", min(len(samples), 6000))
except Exception as e:
    print(f"  gsm8k skip: {e}", flush=True)

# ── 4. Существующий mix (10%) ──
print("=== Existing mix ===", flush=True)
try:
    from data import InstructDataset
    ds = InstructDataset("data/mix", max_samples=4000)
    for i in range(min(4000, len(ds))):
        x, y = ds[i]
        all_x.append(x.tolist()); all_y.append(y.tolist())
    print(f"  existing_mix: {min(4000, len(ds))}", flush=True)
except Exception as e:
    print(f"  existing skip: {e}", flush=True)

# ── shuffle & save ──
random.seed(42)
combined = list(zip(all_x, all_y))
random.shuffle(combined)
all_x, all_y = zip(*combined) if combined else ([], [])

import numpy as np
xs = np.array(all_x, dtype=np.uint32)
ys = np.array(all_y, dtype=np.int32)
xs.tofile(os.path.join(OUT_DIR, "train_x.bin"))
ys.tofile(os.path.join(OUT_DIR, "train_y.bin"))
json.dump({"n": len(xs), "seq_len": SEQ_LEN}, open(os.path.join(OUT_DIR, "meta.json"), "w"))

print(f"\n=== DONE: {len(xs):,} samples ===", flush=True)
print(f"x: {xs.nbytes/1e6:.1f}MB  y: {ys.nbytes/1e6:.1f}MB", flush=True)
print(f"ignore: {(ys==IGNORE).mean()*100:.1f}%", flush=True)
