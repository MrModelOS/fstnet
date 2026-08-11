#!/usr/bin/env python3
import os, json, random
from datasets import load_dataset
from tokenizers import Tokenizer

TOK = "tokenizer/fst_bpe.json"
OUT_DIR = "data/fc"
SEQ_LEN = 1024
IM_S, IM_E = "<|im_start|>", "<|im_end|>"
PAD, IGNORE = 0, -100

os.makedirs(OUT_DIR, exist_ok=True)
tok = Tokenizer.from_file(TOK)
tok.no_truncation(); tok.no_padding()

def encode_msg(role, content):
    line = IM_S + role + "\n" + content + IM_E
    return tok.encode(line).ids + tok.encode("\n").ids

def parse_glaive(d):
    ids = []
    sys_text = d.get("system", "")
    if sys_text:
        ids += encode_msg("system", sys_text)
    chat = d.get("chat", "")
    role = "user"
    for part in chat.split("\n"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("USER:"):
            role = "user"
            part = part[len("USER:"):].strip()
        elif part.startswith("ASSISTANT:"):
            role = "assistant"
            part = part[len("ASSISTANT:"):].strip()
        elif part.startswith("TOOL:"):
            role = "tool"
            part = part[len("TOOL:"):].strip()
        ids += encode_msg(role, part)
    return ids

def to_sample(ids):
    x, y = [], []
    for i in range(SEQ_LEN):
        j = i + 1
        x.append(ids[i] if i < len(ids) else PAD)
        y.append(ids[j] if j < len(ids) else IGNORE)
    return x, y

print("Loading glaive-function-calling-v2...", flush=True)
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
all_x, all_y = [], []
added = 0
for d in ds:
    try:
        ids = parse_glaive(d)
    except Exception:
        continue
    if len(ids) < 32 or len(ids) > SEQ_LEN:
        continue
    x, y = to_sample(ids)
    all_x.append(x); all_y.append(y)
    added += 1
    if added >= 12000:
        break
    if added % 1000 == 0:
        print(f"  {added}", flush=True)

import numpy as np
combined = list(zip(all_x, all_y))
random.shuffle(combined)
all_x, all_y = zip(*combined)
xs = np.array(all_x, dtype=np.uint32)
ys = np.array(all_y, dtype=np.int32)
xs.tofile(os.path.join(OUT_DIR, "train_x.bin"))
ys.tofile(os.path.join(OUT_DIR, "train_y.bin"))
json.dump({"n": len(xs), "seq_len": SEQ_LEN}, open(os.path.join(OUT_DIR, "meta.json"), "w"))
print(f"DONE: {len(xs)} samples", flush=True)
