#!/usr/bin/env python3
"""pack_data_to_git.py — сжатие датасетов для хранения в git.

Источник истины датасетов — репозиторий: colab_run_full.py распаковывает
data/<name>.json.gz, если файла нет локально и на Диске. Это избавляет от
долгой генерации (200K-500K сэмплов) в каждой сессии.

Запуск (в Colab после сборки data/jarvis_full.json и jarvis_special.json):
  python brain/pack_data_to_git.py
  git add -f data/*.json.gz
  git commit -m "data: jarvis_full + jarvis_special (gz-архивы)"
  git push

GitHub держит файлы до 100MB; большие архивы можно разбить на части
(--split 90MB). colab_run_full.py тогда распакует части автоматически.
"""
import os
import sys
import gzip
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def pack(src, dst):
    n_in = os.path.getsize(src)
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        while True:
            b = fi.read(8 << 20)
            if not b:
                break
            fo.write(b)
    n_out = os.path.getsize(dst)
    print(f"{os.path.basename(src):<28} {n_in/1e6:7.0f}MB -> {n_out/1e6:6.0f}MB")


def split_file(path, max_bytes, chunk=64 << 20):
    """Делит файл на data/<name>.gz.partNN (без сжатия — уже .gz)."""
    n = 0
    with open(path, "rb") as f:
        while True:
            part = f.read(max_bytes)
            if not part:
                break
            n += 1
            dst = f"{path}.part{n:02d}"
            with open(dst, "wb") as fo:
                fo.write(part)
            print(f"  part {n:02d}: {os.path.basename(dst)} ({len(part)/1e6:.0f}MB)")
    os.remove(path)
    print(f"  удалён оригинал {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0,
                    help="макс. размер части в MB (для >100MB архивов, напр. 90)")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    targets = [os.path.join(DATA, n) for n in
               ("jarvis_full.json", "jarvis_special.json")]
    made = 0
    for src in targets:
        if not os.path.exists(src):
            alt = os.path.join(ROOT, "brain", "data", os.path.basename(src))
            if os.path.exists(alt):
                src = alt
        if not os.path.exists(src) or os.path.getsize(src) < 1_000_000:
            print(f"пропуск (нет или пуст): {os.path.basename(src)}")
            continue
        gz = os.path.join(DATA, os.path.basename(src) + ".gz")
        pack(src, gz)
        if args.split and os.path.getsize(gz) > args.split * 1_000_000:
            split_file(gz, args.split * 1_000_000)
        made += 1

    print("\nДальше:")
    print("  git add -f data/*.json.gz*")
    print("  git commit -m 'data: jarvis датасеты (gz)'")
    print("  git push")
    if not made:
        sys.exit(1)


if __name__ == "__main__":
    main()
