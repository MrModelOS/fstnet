"""Google Drive persistence для Colab-обучения.

- Монтирует /content/drive (если запуск в Colab).
- Возвращает директорию для чекпоинтов: /content/drive/MyDrive/fstnet/checkpoints
  либо локальную checkpoints, если Диск недоступен (локальный запуск).
- Переопределение: env FSTNET_CKPT_DIR (например через os.environ).

Использование:
    from colab_drive import setup_checkpoint_dir
    CKPT_DIR = setup_checkpoint_dir(subdir="152m")   # -> .../checkpoints/152m
    best = os.path.join(CKPT_DIR, "best.pt")
"""
import os
import sys
import shutil


def log(msg):
    print(msg, flush=True)


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive():
    """Монтирует Google Диск в /content/drive (только в Colab)."""
    if not _in_colab():
        return "/content/drive"  # вернём строку, но она не будет использована
    drive_root = "/content/drive"
    if os.path.isdir(os.path.join(drive_root, "MyDrive")):
        log("Google Drive уже смонтирован.")
        return drive_root
    try:
        from google.colab import drive
        log("Монтирую Google Диск... (разреши доступ в браузере)")
        drive.mount(drive_root)
        log("Диск смонтирован.")
    except Exception as e:
        log(f"[WARN] Не удалось смонтировать Диск: {e}")
        log("        Продолжаю с локальной папки checkpoints (веса пропадут при вылете!).")
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