"""Сборка смешанного датасета (Data Mixture) для финального дообучения.

Пропорции (по токенам):
  50% - код (CodeAlpaca)
  35% - чат/диалоги (everyday-conversations + systemchats из SmolTalk)
  15% - цепочки рассуждений CoT (GSM8K)

Формат ChatML, loss только по ответу ассистента.
Выход: data/mix/{train_x.bin, train_y.bin, meta.json}
"""
import os
import json
import random
from tokenizers import Tokenizer

SRC_CODE = "/tmp/opencode/codealpaca.json"
SRC_EVERYDAY = "/tmp/opencode/everyday_ds"
SRC_SYSTEMCHATS = "/tmp/opencode/systemchats_ds"
SRC_GSM8K = "/tmp/opencode/gsm8k_ds"
TOK = "tokenizer/fst_bpe.json"
OUT_DIR = "data/mix"
SEQ_LEN = 256

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = 0
IGNORE = -100


def load_ds(path):
    from datasets import load_from_disk
    return load_from_disk(path)


def tokens_for(text: str, tok) -> list:
    return tok.encode(text).ids


def format_chat(messages, tok) -> tuple:
    """messages: [(role, content)]. Возвращает (ids, target_mask),
    где target_mask[i] == True если токен i принадлежит ответу ассистента."""
    ids = []
    mask = []
    im = tok.token_to_id(IM_START)
    im_end = tok.token_to_id(IM_END)
    role_prefix_toks = lambda r: [im] + tok.encode(r).ids
    end_toks = [im_end]

    for role, content in messages:
        is_assistant = role == "assistant"
        prefix = role_prefix_toks(role)
        content_ids = tok.encode(content).ids
        ids.extend(prefix)
        mask.extend([False] * len(prefix))
        ids.extend(content_ids)
        mask.extend([is_assistant] * len(content_ids))
        ids.extend(end_toks)
        mask.extend([False] * len(end_toks))
        ids.append(tok.token_to_id("\n") if tok.token_to_id("\n") is not None else 204)
        mask.append(False)
    return ids, mask


def to_sample(ids, target_mask, tok) -> tuple:
    """ids -> (x, y) с next-token сдвигом и маской по ответу ассистента."""
    x, y = [], []
    for i in range(SEQ_LEN):
        j = i + 1
        x.append(ids[i] if i < len(ids) else PAD)
        y.append(ids[j] if (j < len(ids) and j < len(target_mask) and target_mask[j]) else IGNORE)
    return x, y


def main():
    random.seed(42)
    tok = Tokenizer.from_file(TOK)
    tok.no_truncation()
    tok.no_padding()

    print("Loading sources...")
    code_data = json.load(open(SRC_CODE))
    everyday = load_ds(SRC_EVERYDAY)
    systemchats = load_ds(SRC_SYSTEMCHATS)
    gsm8k = load_ds(SRC_GSM8K)
    print(f"code={len(code_data)} everyday={len(everyday)} systemchats={len(systemchats)} gsm8k={len(gsm8k)}")

    # --- 1. Код: CodeAlpaca (50%) ---
    code_samples = []
    for s in code_data:
        inst = (s.get("instruction") or "").strip()
        inp = (s.get("input") or "").strip()
        out = (s.get("output") or "").strip()
        if not inst or not out:
            continue
        prompt = f"{inst}\n{inp}" if inp else inst
        ids, mask = format_chat([("user", prompt), ("assistant", out)], tok)
        if len(ids) > SEQ_LEN:
            continue
        x, y = to_sample(ids, mask, tok)
        code_samples.append((x, y, len(ids)))
    random.shuffle(code_samples)
    print(f"code samples: {len(code_samples)}")

    # --- 2. Чат (35%) ---
    chat_samples = []
    for d in list(everyday) + list(systemchats):
        msgs = [(m["role"], m["content"]) for m in d["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) < 2:
            continue
        # если длинный диалог — берём хвост (последние сообщения), чтобы влез в SEQ_LEN
        while msgs:
            ids, mask = format_chat(msgs, tok)
            if len(ids) <= SEQ_LEN:
                x, y = to_sample(ids, mask, tok)
                chat_samples.append((x, y, len(ids)))
                break
            msgs = msgs[1:]
    random.shuffle(chat_samples)
    print(f"chat samples: {len(chat_samples)}")

    # --- 3. CoT GSM8K (15%) ---
    cot_samples = []
    for d in gsm8k:
        q = d["question"].strip()
        a = d["answer"].strip()
        # формат с проговариванием шагов уже встроен в GSM8K
        ids, mask = format_chat([("user", q), ("assistant", a)], tok)
        if len(ids) > SEQ_LEN:
            continue
        x, y = to_sample(ids, mask, tok)
        cot_samples.append((x, y, len(ids)))
    random.shuffle(cot_samples)
    print(f"cot samples: {len(cot_samples)}")

    # --- Смешивание: 50/35/15 по числу примеров ---
    # таргет ~1500 шагов * eff batch 16 = 24000 примеров
    target_total = 24000
    target_code = int(target_total * 0.50)
    target_chat = int(target_total * 0.35)
    target_cot = target_total - target_code - target_chat

    code_picked = code_samples[:target_code]
    chat_picked = chat_samples[:target_chat]
    cot_picked = cot_samples[:target_cot]

    all_samples = code_picked + chat_picked + cot_picked
    random.shuffle(all_samples)

    print(f"picked: code={len(code_picked)} chat={len(chat_picked)} "
          f"cot={len(cot_picked)} | total={len(all_samples)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    import numpy as np
    xs = np.array([s[0] for s in all_samples], dtype=np.uint32)
    ys = np.array([s[1] for s in all_samples], dtype=np.int32)
    xs.tofile(os.path.join(OUT_DIR, "train_x.bin"))
    ys.tofile(os.path.join(OUT_DIR, "train_y.bin"))
    json.dump({"n": len(all_samples), "seq_len": SEQ_LEN},
              open(os.path.join(OUT_DIR, "meta.json"), "w"))

    print(f"Wrote data/mix: {len(all_samples):,} samples | "
          f"x {xs.nbytes/1e6:.0f}MB y {ys.nbytes/1e6:.0f}MB")
    print(f"ignore ratio: {(ys == IGNORE).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
