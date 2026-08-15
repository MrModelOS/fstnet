#!/usr/bin/env python3
"""export_1bit.py — упаковка обученного чекпоинта в 1-bit дистрибутив.

Берёт чекпоинт (лучше moF_best_1bit.pt после bit_field_tune.py) и пакует:
  - BitLinear.weight     -> sign(w) упакован в uint32 (1 bit/вес) + scale fp16
  - ContinuousField U/V  -> sign(U)/sign(V) в uint32 + scale fp16
  - эмбеддинг/hypernet/RMSNorm -> fp16 (не бинаризуются архитектурой)

Итог ~543MB для 3.4B (BitLinear ~213MB + поля ~200MB + прочее fp16).

Запуск:
  FSTNET_STAGE=1 python jarvis_engine/trainer/export_1bit.py
  -> checkpoints/3b_mof/model_1bit.pt (+ .part на Диск)
  FSTNET_STAGE=2 ... -> checkpoints/3b_mof_stage2/model_1bit.pt

Выходной файл:
  {
    "config": FSTMoFConfig,
    "packed": { "<name>": {"bits": uint32 tensor (..., 32x меньше), "scale": fp16 tensor} },
    "fp16":   { "<name>": fp16 tensor },        # embedding, hypernet, norms
    "quant":  {"binarize": 1.0, "fields": True}
  }

Для инференса: load_1bit(model, state) в этом же файле раскладывает обратно.
"""
import os
import sys
import shutil

import torch
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.dirname(_THIS))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "brain"))
sys.path.insert(0, os.path.join(_ROOT, "brain", "model"))

from config_3b_mof import FSTMoFConfig
from colab_drive import setup_checkpoint_dir, load_checkpoint


def log(m): print(m, flush=True)


def pack_sign(t: torch.Tensor) -> torch.Tensor:
    """sign(t) -> uint32, 32 знака в одном слове (LSB-first). Вход: float CPU."""
    s = torch.sign(t)
    s = (s + 1) // 2                      # -1 -> 0, +1 -> 1
    s = s.reshape(-1).to(torch.int64)
    n = s.numel()
    pad = (-n) % 32
    if pad:
        s = torch.cat([s, torch.zeros(pad, dtype=s.dtype)])
    bits = s.reshape(-1, 32) << torch.arange(32, device=s.device).reshape(1, 32)
    return bits.sum(dim=1).to(torch.uint32)   # (n/32,)


def unpack_sign(bits: torch.Tensor, shape, device=None, dtype=torch.float32):
    """Обратно: uint32 -> float знаки {-1,+1} формы shape."""
    b = bits.to(torch.int64).reshape(-1, 1)
    s = ((b >> torch.arange(32, device=b.device)) & 1).reshape(-1)
    n = int(np.prod(shape))
    s = s[:n]
    return (s * 2 - 1).to(dtype).reshape(shape).to(device)


def row_scale(t: torch.Tensor, dim=1):
    """Как в forward: mean|.| по dim=1 (для U/V: (n,1,r)/(n,1,o))."""
    return t.abs().mean(dim=dim, keepdim=True).clamp_min(1e-12)


def pack_model(state, cfg: FSTMoFConfig):
    packed, fp16 = {}, {}
    quant_bits = 0
    quant_head = bool(getattr(cfg, "quant_head", False))
    for name, t in state.items():
        if name.endswith(".weight") and (".W0" in name or ".Wq" in name
                or ".Wk" in name or ".Wv" in name or ".Wo" in name) \
                or (name == "head.weight" and quant_head):
            # BitLinear
            w = t.float()
            s = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
            packed[name] = {"bits": pack_sign(w), "scale": s.half(), "shape": tuple(w.shape)}
            quant_bits += t.numel()
        elif name.endswith(".U") or name.endswith(".V"):
            # ContinuousField U/V: scale как в forward (dim=1)
            w = t.float()
            s = row_scale(w, dim=1)
            packed[name] = {"bits": pack_sign(w), "scale": s.half(), "shape": tuple(w.shape)}
            quant_bits += t.numel()
        elif ".scale" in name:
            fp16[name] = t.half()
        else:
            fp16[name] = t.half()
    return packed, fp16, quant_bits


def bytes_of(packed, fp16):
    b = 0
    for d in packed.values():
        b += d["bits"].numel() * 4 + d["scale"].numel() * 2
    for t in fp16.values():
        b += t.numel() * 2
    return b


def load_1bit(model, packed, fp16, device=None, dtype=torch.float32):
    """Раскладывает 1-bit дистрибутив в state_dict модели (для инференса)."""
    sd = {}
    for name, t in fp16.items():
        sd[name] = t.to(dtype)
    for name, d in packed.items():
        shape = tuple(d["shape"])
        w = unpack_sign(d["bits"], shape, dtype=dtype) * d["scale"].to(dtype)
        sd[name] = w
    model.load_state_dict(sd, strict=False)
    if device:
        model.to(device)
    return model


def main():
    STAGE = int(os.environ.get("FSTNET_STAGE", "1"))
    subdir = "3b_mof" if STAGE == 1 else "3b_mof_stage2"
    ckpt_dir = setup_checkpoint_dir(subdir=subdir)

    # приоритет: 1bit (после tune) -> обычный best
    srcs = [os.path.join(ckpt_dir, "moF_best_1bit.pt"),
            os.path.join(ckpt_dir, "moF_best.pt"),
            os.path.join("/content", f"best_3b_mof{'' if STAGE == 1 else '_stage2'}.pt")]
    src = next((s for s in srcs if os.path.exists(s)), None)
    if not src:
        log(f"[FAIL] чекпоинт не найден (искал {srcs})")
        sys.exit(1)

    log(f"Loading {src}")
    ck = load_checkpoint(src)
    sd = ck["model_state"]
    cfg = ck.get("config") or FSTMoFConfig()

    # строгий прогон: имена в чекпоинте от torch.compile могут иметь _orig_mod.
    if any("_orig_mod." in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # печатаем распределение по типам
    n_bit, n_other = 0, 0
    for k, v in sd.items():
        if (k.endswith(".weight") and (".W0" in k
                or (k == "head.weight" and getattr(cfg, "quant_head", False))
                or ".Wq" in k or ".Wk" in k or ".Wv" in k or ".Wo" in k)) \
                or k.endswith(".U") or k.endswith(".V"):
            n_bit += v.numel()
        else:
            n_other += v.numel()
    log(f"Параметры: 1-bit {n_bit/1e9:.2f}B, fp16 {n_other/1e9:.2f}B, "
        f"итого {(n_bit+n_other)/1e9:.2f}B")

    packed, fp16, quant_bits = pack_model(sd, cfg)
    nbytes = bytes_of(packed, fp16)
    log(f"1-bit весов: {quant_bits/1e9:.2f}B -> упаковано, "
        f"суммарный размер: {nbytes/1e6:.0f}MB")

    out = {"config": cfg, "packed": packed, "fp16": fp16,
           "quant": {"binarize": 1.0, "fields": True}}
    local = os.path.join("/content", f"model_1bit{'' if STAGE == 1 else '_stage2'}.pt")
    torch.save(out, local)
    log(f"Saved {local} ({os.path.getsize(local)/1e6:.0f}MB)")

    drive = os.path.join(ckpt_dir, "model_1bit.pt")
    try:
        os.makedirs(os.path.dirname(drive), exist_ok=True)
        shutil.copyfile(local, drive + ".part")
        os.replace(drive + ".part", drive)
        log(f"-> Диск: {drive}")
    except Exception as e:
        log(f"[WARN] Диск: {e}")


if __name__ == "__main__":
    main()
