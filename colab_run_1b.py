#!/usr/bin/env python3
"""Полный авто-запуск обучения FST-Net 1B 1-bit MoF в Colab.

Одна ячейка (копируй целиком):
  !wget -q https://raw.githubusercontent.com/MrModelOS/fstnet/master/colab_run_1b.py
  %run colab_run_1b.py

Важно: запускай через %run (не !python) — тогда монтирование Диска выполняется
в ядре ноутбука и покажет ссылку авторизации прямо в ячейке.

Что делает скрипт:
  1. Монтирует Google Диск.
  2. Обеспечивает репозиторий в /content/fstnet.
  3. Ставит зависимости (tokenizers, tqdm).
  4. Обеспечивает датасет data/jarvis_full.json (с Диска или git-архива).
  5. Запускает train_1b.py (993M params, batch 16×4=64, torch.compile).
  6. Чекпоинты на Диск: MyDrive/fstnet_1b/.
  7. Лог прогона -> Диск.

Env-перезапись:
  RUN_EPOCHS   — кол-во эпох (по умолчанию 2)
  RUN_LR       — learning rate (по умолчанию 3e-4)
  RUN_BATCH    — micro-batch (по умолчанию 16)
  RUN_ACCUM    — accumulation steps (по умолчанию 4)
"""
import os
import sys
import time
import gzip
import shutil
import subprocess

REPO = "https://github.com/MrModelOS/fstnet.git"
SKILL = "/content/fstnet"
DATA = os.path.join(SKILL, "data")
LOG_DIR = os.path.join(SKILL, "logs")
DRIVE_DATA = "/content/drive/MyDrive/fstnet/data"
DRIVE_LOG = "/content/drive/MyDrive/fstnet/logs"
DRIVE_CKPT = "/content/drive/MyDrive/fstnet_1b"

EPOCHS = int(os.environ.get("RUN_EPOCHS", "2"))
LR = os.environ.get("RUN_LR", "3e-4")
BATCH = os.environ.get("RUN_BATCH", "16")
ACCUM = os.environ.get("RUN_ACCUM", "4")


def log(msg):
    print(msg, flush=True)


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def drive_mounted():
    return os.path.isdir("/content/drive/MyDrive")


def mount_drive():
    if not in_colab():
        log("[local] Не Colab — пропускаю монтирование Диска.")
        return
    if drive_mounted():
        log("Google Диск уже смонтирован.")
        return
    try:
        from google.colab import drive
        log("Монтирую Google Диск — в браузере разреши доступ...")
        drive.mount("/content/drive")
    except Exception as e:
        log(f"[WARN] Не удалось смонтировать Диск: {e}")
    if drive_mounted():
        log("Диск смонтирован.")
    else:
        log("=" * 60)
        log("Диск НЕ подключён. Чекпоинты останутся только в /content!")
        log("Запускай через %run — тогда mount отработает в ячейке.")
        log("=" * 60)


def ensure_repo():
    if os.path.isdir(os.path.join(SKILL, ".git")):
        log("Репозиторий уже есть — git pull.")
        subprocess.run(["git", "-C", SKILL, "pull", "--ff-only", "-q"], check=False)
    else:
        if os.path.isdir(SKILL):
            stale = f"{SKILL}.stale.{int(time.time())}"
            os.rename(SKILL, stale)
            log(f"Убрал не-git директорию в {stale}")
        log(f"Клонирую: {REPO}")
        subprocess.run(["git", "clone", "-q", REPO, SKILL], check=True)
    os.chdir(SKILL)
    sys.path.insert(0, SKILL)
    log(f"CWD: {os.getcwd()}")


def pip(*pkgs):
    log(f"pip install: {', '.join(pkgs)}")
    subprocess.run(["pip", "install", "-q", *pkgs, "--break-system-packages"],
                   check=False)


def run(cmd, env, logf, timeout=None):
    log(f"$ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
        if logf:
            logf.write(line)
            logf.flush()
    rc = p.wait(timeout=timeout)
    if rc != 0:
        log(f"[rc={rc}] {cmd[0]} завершился с ошибкой")
    return rc


def sync_file(local, drive):
    if not drive_mounted():
        return
    try:
        os.makedirs(os.path.dirname(drive), exist_ok=True)
        if (not os.path.exists(drive)) or os.path.getsize(drive) != os.path.getsize(local):
            log(f"sync -> {drive}")
            shutil.copyfile(local, drive)
    except Exception as e:
        log(f"[WARN] sync {local} -> {drive}: {e}")


def ensure_data():
    """Возвращает локальный путь к jarvis_full.json."""
    name = "jarvis_full.json"
    local = os.path.join(DATA, name)
    drive = os.path.join(DRIVE_DATA, name)

    if os.path.exists(local) and os.path.getsize(local) > 1_000_000:
        log(f"Датасет уже есть: {local} ({os.path.getsize(local)/1e6:.0f}MB)")
        return local

    if os.path.exists(drive):
        log(f"Копирую с Диска: {drive} -> {local}")
        os.makedirs(DATA, exist_ok=True)
        shutil.copyfile(drive, local)
        return local

    # git-архив
    archive = os.path.join(DATA, name + ".gz")
    parts = sorted(p for p in os.listdir(DATA) if p.startswith(name + ".gz.part"))
    if parts:
        log(f"Склеиваю части: {len(parts)} -> {archive}")
        with open(archive, "wb") as fo:
            for p in parts:
                with open(os.path.join(DATA, p), "rb") as fi:
                    shutil.copyfileobj(fi, fo)
    if os.path.exists(archive):
        log(f"Распаковываю: {archive} -> {local}")
        os.makedirs(DATA, exist_ok=True)
        with gzip.open(archive, "rb") as gz, open(local, "wb") as f:
            shutil.copyfileobj(gz, f)
        log(f"  распакован ({os.path.getsize(local)/1e6:.0f}MB)")
        return local

    raise SystemExit(
        f"[FAIL] Датасет {name} не найден. Скопируй на Диск: "
        f"{DRIVE_DATA}/{name}")


def train_env():
    env = dict(os.environ)
    env["FSTNET_DATA"] = "data/jarvis_full.json"
    env["FSTNET_EPOCHS"] = str(EPOCHS)
    env["FSTNET_LR"] = LR
    env["FSTNET_BATCH"] = BATCH
    env["FSTNET_ACCUM"] = ACCUM
    env["FSTNET_DRIVE_SUB"] = "fstnet_1b"
    if drive_mounted():
        env["FSTNET_CKPT_DIR"] = DRIVE_CKPT
    return env


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")

    log("=" * 60)
    log("FST-NET 1B TRAINING | 1-bit MoF (993M params)")
    log("=" * 60)
    mount_drive()
    ensure_repo()

    os.makedirs(LOG_DIR, exist_ok=True)
    logf = open(os.path.join(LOG_DIR, f"run_1b_{ts}.log"), "w")

    pip("tokenizers", "tqdm")

    ensure_data()

    log(f"\n########## TRAINING 1B ##########")
    log(f"  epochs={EPOCHS} | batch={BATCH} accum={ACCUM} (eff={int(BATCH)*int(ACCUM)}) | lr={LR}")
    log(f"  torch.compile=reduce-overhead | loss-scale x4096")
    log(f"  чекпоинты -> {DRIVE_CKPT}/")

    rc = run(["python", "train_1b.py"], train_env(), logf)

    if rc != 0:
        log(f"[FAIL] Training упал (rc={rc}). Смотри лог выше.")
        sync_file(os.path.join(LOG_DIR, f"run_1b_{ts}.log"),
                  os.path.join(DRIVE_LOG, f"run_1b_{ts}.log"))
        sys.exit(1)

    sync_file(os.path.join(LOG_DIR, f"run_1b_{ts}.log"),
              os.path.join(DRIVE_LOG, f"run_1b_{ts}.log"))

    log("\n" + "=" * 60)
    log("ГОТОВО. Артефакты:")
    log(f"  чекпоинты: {DRIVE_CKPT}/")
    log(f"  лог:       {DRIVE_LOG}/run_1b_{ts}.log")
    log("=" * 60)


if __name__ == "__main__":
    main()
