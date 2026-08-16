"""Google Drive persistence для Colab-обучения.

- Монтирует /content/drive (если запуск в Colab).
- Возвращает директорию для чекпоинтов: /content/drive/MyDrive/fstnet/checkpoints
  либо локальную checkpoints, если Диск недоступен (локальный запуск).
- Переопределение: env FSTNET_CKPT_DIR (например через os.environ).
- save_checkpoint/load_checkpoint: чекпоинты пишутся gzip-сжатыми
  и БЕЗ full-precision весов BitLinear: бинаризованные веса хранятся как
  sign (упакован в uint32) + scale fp16 (см. pack_binarized_state).
  Это даёт ~6-8x меньше, чем fp16, и читается обратно в исходные float.

Использование:
    from colab_drive import setup_checkpoint_dir, save_checkpoint, load_checkpoint
    CKPT_DIR = setup_checkpoint_dir(subdir="152m")   # -> .../checkpoints/152m
    best = os.path.join(CKPT_DIR, "best.pt")
"""
import os
import sys
import gzip
import shutil

import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


# -- Паковка бинаризованных весов ---------------------------------------------
# При binarize=1.0 forward использует sign(W)*scale, где scale=mean|W| по строкам.
# Значит W можно хранить как sign(W) + scale: восстановленный W' = sign(W)*scale
# даёт ТОТ ЖЕ forward, что и исходный W (проверено). На диске это 1 bit/вес
# (вместо 2 байт fp16) + fp16 scale на строку: ~6-8x меньше.
def _is_bitlinear_weight(name):
    """BitLinear/ContinuousField веса, бинаризуемые архитектурой."""
    if name.endswith(".U") or name.endswith(".V"):
        return True
    if not name.endswith(".weight"):
        return False
    return (".W0" in name or ".Wq" in name or ".Wk" in name
            or ".Wv" in name or ".Wo" in name or name == "head.weight")


def _pack_sign(t):
    """sign(t) -> uint32, 32 знака в слове (LSB-first). Вход: float."""
    s = torch.sign(t)
    s = (s + 1) // 2                      # -1 -> 0, +1 -> 1
    s = s.reshape(-1).to(torch.int64)
    n = s.numel()
    pad = (-n) % 32
    if pad:
        s = torch.cat([s, torch.zeros(pad, dtype=s.dtype)])
    bits = s.reshape(-1, 32) << torch.arange(32, device=s.device).reshape(1, 32)
    return bits.sum(dim=1).to(torch.uint32)   # (n/32,)


def _unpack_sign(bits, shape, device=None, dtype=torch.float32):
    """Обратно: uint32 -> float знаки {-1,+1} формы shape."""
    b = bits.to(torch.int64).reshape(-1, 1)
    s = ((b >> torch.arange(32, device=b.device)) & 1).reshape(-1)
    n = int(np.prod(shape))
    s = s[:n]
    return (s * 2 - 1).to(dtype).reshape(shape).to(device)


def pack_binarized_state(model_state, quant_head=False):
    """Возвращает копию state_dict с бинаризованными весами в виде знаков.

    {name: float tensor} -> {name: {"bits": uint32, "scale": fp16, "shape": ...}}
    head.weight пакуется только если quant_head (иначе он fp16 Linear).
    Обратно: unpack_binarized_state().
    """
    packed = {}
    for name, t in model_state.items():
        is_head = name == "head.weight"
        if (not is_head and _is_bitlinear_weight(name)) or (
                is_head and quant_head):
            w = t.float()
            s = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
            packed[name] = {"bits": _pack_sign(w), "scale": s.half(),
                            "shape": tuple(w.shape)}
        else:
            packed[name] = t
    return packed


def unpack_binarized_state(model_state, dtype=torch.float32):
    """Обратно: распаковывает знаки в float веса sign*scale.

    Принимает И packed (с dict-значениями), И обычный state_dict (нетто).
    """
    sd = {}
    for name, v in model_state.items():
        if isinstance(v, dict) and "bits" in v:
            w = _unpack_sign(v["bits"], tuple(v["shape"]), dtype=dtype)
            w = w * v["scale"].to(dtype)
            sd[name] = w
        else:
            sd[name] = v
    return sd


def save_checkpoint(path, state, compresslevel=9, pack=False, quant_head=False):
    """Сохраняет чекпоинт: бинаризованные веса -> знаки, затем gzip.

    pack=True: BitLinear/U/V -> sign+scale, ~6-8x меньше. БЕЗОПАСНО только
    когда модель ПОЛНОСТЬЮ бинаризована (binarize==1.0): при частичной
    бинаризации forward использует сырые веса (1-b)*w + b*sign*scale, и
    паковка их потеряет. Вызывающий код отвечает за передачу pack=True
    лишь при полной бинаризации (см. run_trainer/bit_field_tune).
    quant_head=True: head тоже паковать (он BitLinear в этом режиме).
    pack=False: как раньше, полные fp16.
    """
    if pack and "model_state" in state:
        state = dict(state)
        state["model_state"] = pack_binarized_state(
            state["model_state"], quant_head=quant_head)
    with gzip.open(path, "wb", compresslevel=compresslevel) as f:
        torch.save(state, f)


def load_checkpoint(path, map_location="cpu"):
    """Загружает чекпоинт: gzip с распаковкой знаков ИЛИ обычный."""
    try:
        with gzip.open(path, "rb") as f:
            state = torch.load(f, map_location=map_location, weights_only=False)
    except (OSError, EOFError):
        state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        ms = state["model_state"]
        if any(isinstance(v, dict) and "bits" in v for v in ms.values()):
            state["model_state"] = unpack_binarized_state(ms)
    return state


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive():
    """Монтирует Google Диск в /content/drive (только в Colab).

    drive.mount() требует интерактивной авторизации через живой kernel
    ноутбука. Из скрипта (!python) она может падать, поэтому:
      - если Диск уже смонтирован (в т.ч. ранее) — используем как есть;
      - при неудаче даём явную инструкцию смонтировать ячейкой и продолжить.
    """
    if not _in_colab():
        return "/content/drive"  # строка не будет использоваться
    drive_root = "/content/drive"
    if os.path.isdir(os.path.join(drive_root, "MyDrive")):
        log("Google Drive уже смонтирован.")
        return drive_root
    try:
        from google.colab import drive
        log("Монтирую Google Диск... (разреши доступ в браузере)")
        drive.mount(drive_root)
        if os.path.isdir(os.path.join(drive_root, "MyDrive")):
            log("Диск смонтирован.")
            return drive_root
    except Exception as e:
        log(f"[WARN] Не удалось смонтировать Диск: {e}")
    log("")
    log("=" * 60)
    log("Google Drive НЕ подключён — чекпоинты рискуют пропасть при вылете!")
    log("Смонтируй вручную ячейкой, останови и перезапусти скрипт:")
    log("    from google.colab import drive; drive.mount('/content/drive')")
    log("Затем продолжить (чекпоинты уйдут на Диск).")
    log("=" * 60)
    return drive_root


def setup_checkpoint_dir(subdir=None, base="checkpoints"):
    """Возвращает директорию чекпоинтов, создавая её.

    Приоритет путей:
      1. FSTNET_CKPT_DIR (если задан)
      2. /content/drive/MyDrive/fstnet/checkpoints (если Диск примонтирован)
      3. локальная ./checkpoints
    """
    env_dir = os.environ.get("FSTNET_CKPT_DIR", "").strip()
    drive_root = mount_drive()

    if env_dir:
        ckpt_root = env_dir
    elif _in_colab() and os.path.isdir(os.path.join(drive_root, "MyDrive")):
        ckpt_root = os.path.join(drive_root, "MyDrive", "fstnet", "checkpoints")
        log(f"Чекпоинты на Google Диске: {ckpt_root}")
    else:
        ckpt_root = os.path.abspath(base)
        log(f"Чекпоинты локально: {ckpt_root} (Диск недоступен)")

    if subdir:
        ckpt_root = os.path.join(ckpt_root, subdir)
    os.makedirs(ckpt_root, exist_ok=True)
    return ckpt_root