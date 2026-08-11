"""Потоковая токенизация корпуса в бинарный файл (без накопления в RAM)."""
import os
import numpy as np
from tokenizers import Tokenizer

CORPUS = "data/local/corpus.txt"
TOK_PATH = "tokenizer/fst_bpe.json"
OUT_BIN = "data/local/train.bin"

print("Loading tokenizer...")
tok = Tokenizer.from_file(TOK_PATH)
tok.no_truncation()
tok.no_padding()

total = 0
with open(OUT_BIN, "wb") as out, open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
    # Кодируем по абзацам, чтобы не держать весь файл
    batch = []
    for line in f:
        if line.strip():
            batch.append(line)
        if len(batch) >= 2000:
            ids = tok.encode("\n".join(batch)).ids
            np.array(ids, dtype=np.uint32).tofile(out)
            total += len(ids)
            batch = []
    if batch:
        ids = tok.encode("\n".join(batch)).ids
        np.array(ids, dtype=np.uint32).tofile(out)
        total += len(ids)

print(f"Saved {OUT_BIN} ({os.path.getsize(OUT_BIN)/1e6:.1f} MB)")
print(f"Total tokens: {total:,}")
