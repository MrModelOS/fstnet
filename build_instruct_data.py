"""Подготовка instruction-датасета: CodeAlpaca -> бинарные примеры с масками loss.

Формат (ChatML): <|im_start|>user\nPROMPT<|im_end|>\n<|im_start|>assistant\nOUTPUT<|im_end|>
Loss считается ТОЛЬКО по OUTPUT (y=-100 в user-части).

Выход:
  data/instruct/train_x.bin  (uint32, N x seq_len) - входные токены
  data/instruct/train_y.bin  (int32,  N x seq_len) - таргеты, -100 = ignore
  data/instruct/meta.json    (N, seq_len)
"""
import os
import json
import numpy as np
from tokenizers import Tokenizer

SRC = "/tmp/opencode/codealpaca.json"
TOK = "tokenizer/fst_bpe.json"
OUT_DIR = "data/instruct"
SEQ_LEN = 256
MAX_SAMPLES = 19_000

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = 0
IGNORE = -100


def format_example(inst: str, inp: str, out: str) -> str:
    if inp:
        prompt = f"{inst}\n{inp}"
    else:
        prompt = inst
    return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n{out}{IM_END}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading tokenizer...")
    tok = Tokenizer.from_file(TOK)
    tok.no_truncation()
    tok.no_padding()

    print(f"Loading {SRC}...")
    data = json.load(open(SRC))
    print(f"Samples: {len(data):,}")

    xs, ys = [], []
    skipped_long = 0
    skipped_empty = 0

    for s in data:
        inst = (s.get("instruction") or "").strip()
        inp = (s.get("input") or "").strip()
        out = (s.get("output") or "").strip()
        if not inst or not out:
            skipped_empty += 1
            continue

        text = format_example(inst, inp, out)
        ids = tok.encode(text).ids
        if len(ids) > SEQ_LEN:
            skipped_long += 1
            continue
        if len(ids) < 8:
            skipped_empty += 1
            continue

        # find assistant start: second <|im_start|> token
        starts = [i for i, t in enumerate(ids) if t == tok.token_to_id(IM_START)]
        if len(starts) < 2:
            skipped_empty += 1
            continue
        asst_token_idx = starts[-1]  # <|im_start|> of assistant turn
        # next-token сдвиг: x[i] предсказывает ids[i+1]
        # loss только на таргетах из ответа ассистента (i+1 >= asst_token_idx)
        x = []
        y = []
        for i in range(SEQ_LEN):
            j = i + 1
            if i < len(ids):
                x.append(ids[i])
            else:
                x.append(PAD)
            if j >= asst_token_idx and j < len(ids):
                y.append(ids[j])
            else:
                y.append(IGNORE)

        x = np.array(x, dtype=np.uint32)
        y = np.array(y, dtype=np.int32)
        assert len(x) == len(y) == SEQ_LEN
        xs.append(x)
        ys.append(y)

        if len(xs) >= MAX_SAMPLES:
            break

    print(f"Kept: {len(xs):,} | skipped_long: {skipped_long} | skipped_empty: {skipped_empty}")

    x_arr = np.stack(xs)
    y_arr = np.stack(ys)

    x_arr.astype(np.uint32).tofile(os.path.join(OUT_DIR, "train_x.bin"))
    y_arr.astype(np.int32).tofile(os.path.join(OUT_DIR, "train_y.bin"))
    json.dump({"n": len(xs), "seq_len": SEQ_LEN}, open(os.path.join(OUT_DIR, "meta.json"), "w"))

    print(f"train_x.bin: {x_arr.nbytes/1e6:.1f} MB, train_y.bin: {y_arr.nbytes/1e6:.1f} MB")
    print(f"IGNORE ratio: {(y_arr == IGNORE).mean()*100:.1f}% (user parts)")


if __name__ == "__main__":
    main()
