#!/usr/bin/env python3
"""Фетч готового датасета OpenHermes-2.5 для обучения MoF FST-Net (JARVIS Core).

Что делает:
  - стримит teknium/OpenHermes-2.5 (~1M примеров) с HF (не качает всё сразу)
  - фильтрует: <50 токенов ответа и дубликаты — долой
  - конвертирует ShareGPT -> наш формат [[role, content], ...] (ChatML-совместимо)
  - пишет data/jarvis_openhermes.json (аналог выхода build_jarvis_data.py)

Итоговый пайплайн датасета (мерж):
  OpenHermes (массовая база диалога) + синтетика JARVIS (персона/tool-call/Сэр)
    -> data/jarvis_full.json  (для jarvis_engine/trainer/run_trainer.py)

Запуск (локально или Colab):
  !pip install -q datasets
  !python fetch_openhermes.py --max-examples 300000 --min-tokens 50 --out data/jarvis_openhermes.json
  !python merge_datasets.py --add data/jarvis_openhermes.json --merge data/jarvis_full.json
Env: OVH_MAX, OVH_MIN_TOK, OVH_SEED
"""
import os
import sys
import json
import time
import hashlib
import argparse

def log(msg): print(msg, flush=True)

DATASET = "teknium/OpenHermes-2.5"


def conv_to_ours(conversation):
    """ShareGPT: [{"from": "system|human|gpt", "value": "..."}] -> [[role, content], ...]"""
    roles = {"system": "system", "human": "user", "gpt": "assistant"}
    out = []
    for turn in conversation or []:
        r = roles.get(turn.get("from"))
        if not r:
            continue
        v = (turn.get("value") or "").strip()
        if not v:
            continue
        out.append((r, v))
    return out if out else None


def iter_jsonl(path):
    """Потоково читает JSON-массив объектов (конверсации) — без загрузки всего в RAM."""
    import ijson
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            yield obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=int(os.environ.get("OVH_MAX", "300000")))
    ap.add_argument("--min-tokens", type=int, default=int(os.environ.get("OVH_MIN_TOK", "50")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("OVH_SEED", "42")))
    ap.add_argument("--out", default="data/jarvis_openhermes.json")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--json-file", default="")
    args = ap.parse_args()

    from datasets import load_dataset
    if args.json_file:
        import ijson
        log(f"Читаю локальный JSON: {args.json_file}")
        raw = iter_jsonl(args.json_file)
    elif args.parquet:
        import pandas as pd
        log(f"Читаю локальный parquet: {args.parquet}")
        df = pd.read_parquet(args.parquet)
        total = len(df)
        log(f"Всего строк в parquet: {total}")
        def gen():
            for row in df.itertuples(index=False):
                d = row._asdict() if hasattr(row, "_asdict") else row
                conv = d.get("conversations") or d.get("conversation")
                yield {"conversations": conv}
        raw = gen()
    else:
        log(f"Стримлю {DATASET} (max {args.max_examples})...")
        raw = load_dataset(DATASET, split="train", streaming=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen, rows, skipped = set(), [], 0
    t0 = time.time()
    for i, sample in enumerate(raw):
        if len(rows) >= args.max_examples:
            break
        conv = conv_to_ours(sample.get("conversations") or sample.get("conversation") or [])
        if not conv:
            skipped += 1
            continue
        merged = " ".join(c for _, c in conv)
        if len(merged) < args.min_tokens * 4:
            skipped += 1
            continue
        h = hashlib.sha1(merged.encode()).hexdigest()
        if h in seen:
            skipped += 1
            continue
        seen.add(h)
        rows.append(conv)
        if (i + 1) % 50000 == 0:
            el = time.time() - t0
            log(f"  scanned {i+1} | kept {len(rows)} | skip {skipped} | {el/60:.1f}min")

    with open(args.out, "w") as f:
        json.dump(rows, f)
    log(f"OK: {len(rows)} конверсаций -> {args.out} (сканировано ~{i+1}, пропущено {skipped})")


if __name__ == "__main__":
    main()