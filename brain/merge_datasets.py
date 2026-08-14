#!/usr/bin/env python3
"""Мерж датасетов для train_colab_mof.py в единый data/jarvis_full.json.

Стратегия: добавляемый датасет (OpenHermes = массовая диалоговая база) идёт
вперемешку с базой (JARVIS синтетика = персона/tool-call/Сэр). Пропорция
регулируется --weight: пары (base, add) перемешиваются с весом add.
"""
import os
import json
import random
import argparse


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", default="data/jarvis_openhermes.json",
                    help="добавляемый датасет (массовый диалог)")
    ap.add_argument("--merge", default="data/jarvis_full.json",
                    help="база (синтетика JARVIS), которую мержим И перезаписываем")
    ap.add_argument("--weight", type=float, default=0.7,
                    help="доля добавляемого датасета в итоге (0.7 = смесь)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = load(args.merge)
    add = load(args.add)
    rng = random.Random(args.seed)

    target_add = int(len(base) * args.weight / max(1 - args.weight, 1e-9))
    target_add = min(target_add, len(add))
    picked = rng.sample(add, target_add) if target_add < len(add) else add[:]

    merged = base + picked
    rng.shuffle(merged)

    with open(args.merge, "w") as f:
        json.dump(merged, f)
    print(f"base {len(base)} + add {len(picked)} ({args.weight:.0%} of total) "
          f"= {len(merged)} -> {args.merge}")


if __name__ == "__main__":
    main()