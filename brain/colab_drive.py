"""Google Drive persistence для Colab-обучения.

- Монтирует /content/drive (если запуск в Colab).
- Возвращает директорию для чекпоинтов: /content/drive/MyDrive/fstnet/checkpoints
  либо локальную checkpoints, если Диск недоступен (локальный запуск).
- Переопределение: env FSTNET_CKPT_DIR (например через os.environ).
- save_checkpoint/load_checkpoint: чекпоинты пишутся gzip-сжатыми
  (~2-3x меньше, быстрее запись в /content), читаются оба формата.

Использование:
    from colab_drive import setup_checkpoint_dir, save_checkpoint, load_checkpoint
    CKPT_DIR = setup_checkpoint_dir(subdir="152m")   # -> .../checkpoints/152m
    best = os.path.join(CKPT_DIR, "best.pt")
"""
import os
import sys
import gzip
import shutil


def log(msg):
    print(msg, flush=True)


def save_checkpoint(path, state, compresslevel=3):
    """Сохраняет чекпоинт gzip-сжатым (torch.save -> gzip файл).

    Уменьшает размер в 2-3 раза: на T4 чекпоинт 3.3B параметров fp16
    ~6.6GB -> ~2-2.5GB, и запись в /content быстрее.
    """
    import torch
    with gzip.open(path, "wb", compresslevel=compresslevel) as f:
        torch.save(state, f)


def load_checkpoint(path, map_location="cpu"):
    """Загружает чекпоинт: gzip-сжатый ИЛИ обычный (для старых файлов)."""
    import torch
    try:
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, weights_only=False)
    except (OSError, EOFError):
        return torch.load(path, map_location=map_location, weights_only=False)


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