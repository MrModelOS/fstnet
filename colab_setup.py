"""Bootstrap для Colab: монтирует Диск, клонирует/пуллит fstnet, создаёт каталоги.

Запуск в Colab (одна ячейка):
  !python colab_setup.py

Идемпотентный: повторный запуск не ломает ничего, лишь подтягивает изменения.
"""
import os
import sys
import subprocess

REPO = "https://github.com/MrModelOS/fstnet.git"
SKILL_DIR = "/content/fstnet"

def log(msg):
    print(msg, flush=True)

def run(cmd):
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive():
    if not in_colab():
        log("Не Colab — пропускаю монтирование Диска.")
        return
    from google.colab import drive
    root = "/content/drive"
    if os.path.isdir(os.path.join(root, "MyDrive")):
        log("Google Drive уже смонтирован.")
        return
    log("Монтирую Google Диск... (разреши доступ в браузере)")
    try:
        drive.mount(root)
        if os.path.isdir(os.path.join(root, "MyDrive")):
            log("Диск смонтирован.")
    except Exception as e:
        log(f"[WARN] Не удалось смонтировать Диск: {e}")
        log("        Продолжаю локально (в /content — веса пропадут при вылете).")


def sync_repo():
    if os.path.isdir(os.path.join(SKILL_DIR, ".git")):
        log(f"Репозиторий уже есть — делаю git pull: {SKILL_DIR}")
        run(["git", "-C", SKILL_DIR, "pull", "--ff-only"])
    else:
        log(f"Клонирую репозиторий: {SKILL_DIR}")
        run(["git", "clone", REPO, SKILL_DIR])


def make_checkpoints():
    """Создаёт каталоги чекпоинтов локально и (при наличии Диска) на Диске."""
    local = os.path.join(SKILL_DIR, "checkpoints")
    os.makedirs(os.path.join(local, "152m"), exist_ok=True)
    os.makedirs(os.path.join(local, "800m"), exist_ok=True)
    log(f"Каталоги локально: {local}/{{152m,800m}}")

    drive_ckpt = "/content/drive/MyDrive/fstnet/checkpoints"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(os.path.join(drive_ckpt, "152m"), exist_ok=True)
        os.makedirs(os.path.join(drive_ckpt, "800m"), exist_ok=True)
        log(f"Каталоги на Диске: {drive_ckpt}/{{152m,800m}}")
    return local, (drive_ckpt if os.path.isdir("/content/drive/MyDrive") else None)


def main():
    log("=== Colab bootstrap fstnet ===")
    mount_drive()
    sync_repo()
    make_checkpoints()
    log("=== Готово. Дальше: %cd /content/fstnet && pip install ... && python train_... ===")


if __name__ == "__main__":
    main()