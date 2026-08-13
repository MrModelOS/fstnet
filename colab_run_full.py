#!/usr/bin/env python3
"""Полный авто-запуск обучения JARVIS (Stage 1 + Stage 2) в Colab.

Одна ячейка (копируй целиком):
  !git clone -q https://github.com/MrModelOS/fstnet.git /content/fstnet && cd /content/fstnet && python colab_run_full.py

Что делает скрипт:
  1. Монтирует Google Диск (нужно разрешить в браузере; если нет — продолжает локально).
  2. Обеспечивает репозиторий в /content/fstnet (git pull, если уже есть).
  3. Ставит зависимости (tokenizers, tqdm, ijson; datasets — только для OpenHermes).
  4. Датасеты data/jarvis_full.json (Stage 1) и data/jarvis_special.json (Stage 2):
       локально есть -> берём; нет -> копируем с Диска MyDrive/fstnet/data/;
       нет нигде -> собираем (синтетика + OpenHermes для Stage 1, синтетика 500K для Stage 2);
       после обеспечения — синхронизируем на Диск (следующие запуски без сборки).
  5. Stage 1:  FSTNET_EPOCHS=4 FSTNET_LR=2e-4 python train_colab_mof.py
               -> checkpoints/3b_mof/moF_best.pt (локально и на Диске).
  6. Stage 2:  FSTNET_STAGE=2 FSTNET_DATA=data/jarvis_special.json python train_colab_mof.py
               -> checkpoints/3b_mof_stage2/.
  7. Лог прогона -> Диск: MyDrive/fstnet/logs/run_<ts>.log.

Флаги:
  --stage1-only   только Stage 1
  --skip-data     не трогать датасеты (обучение на том, что уже в data/)
  --fresh         удалить локальные чекпоинты/кэши перед стартом (Диск не трогаем)

Env-перезапись (передаются в обучение): RUN_EPOCHS (4), RUN_LR (2e-4),
JARVIS_COUNT (200K синтетики Stage 1), JARVIS_COUNT_SPEC (500K Stage 2),
плюс все FSTNET_*/JARVIS_* уже заданные в ячейке пробрасываются как есть.
"""
import os
import sys
import time
import shutil
import subprocess

REPO = "https://github.com/MrModelOS/fstnet.git"
SKILL = "/content/fstnet"
DATA = os.path.join(SKILL, "data")
LOG_DIR = os.path.join(SKILL, "logs")
DRIVE_DATA = "/content/drive/MyDrive/fstnet/data"
DRIVE_LOG = "/content/drive/MyDrive/fstnet/logs"
DRIVE_CKPT = "/content/drive/MyDrive/fstnet/checkpoints"

EPOCHS = int(os.environ.get("RUN_EPOCHS", "4"))
LR = os.environ.get("RUN_LR", "2e-4")


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
        log("Монтирую Google Диск — в браузере разреши доступ и вставь токен...")
        drive.mount("/content/drive")
    except Exception as e:
        log(f"[WARN] Не удалось смонтировать Диск: {e}")
    if drive_mounted():
        log("Диск смонтирован.")
    else:
        log("=" * 60)
        log("Диск НЕ подключён. Чекпоинты останутся только в /content и "
            "пропадут при вылете рантайма!")
        log("Смонтируй ячейкой и перезапусти:")
        log("    from google.colab import drive; drive.mount('/content/drive')")
        log("=" * 60)


def ensure_repo():
    if not os.path.isdir(os.path.join(SKILL, ".git")):
        log(f"Клонирую репозиторий: {REPO}")
        subprocess.run(["git", "clone", "-q", REPO, SKILL], check=True)
    else:
        log("Репозиторий уже есть — git pull.")
        subprocess.run(["git", "-C", SKILL, "pull", "--ff-only", "-q"], check=False)
    os.chdir(SKILL)
    sys.path.insert(0, SKILL)
    log(f"CWD: {os.getcwd()}")


def pip(*pkgs):
    log(f"pip install: {', '.join(pkgs)}")
    subprocess.run(["pip", "install", "-q", *pkgs, "--break-system-packages"],
                   check=False)


def run(cmd, env, logf, timeout=None):
    """Запуск с потоковым выводом в консоль И в лог-файл."""
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
        log(f"[WARN] Не удалось скопировать {local} -> {drive}: {e}")


def ensure_data(name, drive, build, local_ok=True):
    """Возвращает локальный путь к датасету (из repo/Диска/сборки), синхронизирует на Диск."""
    local = os.path.join(DATA, name)
    if local_ok and os.path.exists(local) and os.path.getsize(local) > 1_000_000:
        log(f"Датасет уже есть: {local} ({os.path.getsize(local)/1e6:.0f}MB)")
    elif os.path.exists(drive):
        log(f"Копирую с Диска: {drive} -> {local}")
        os.makedirs(DATA, exist_ok=True)
        shutil.copyfile(drive, local)
    else:
        log(f"Датасета {name} нет нигде — собираю.")
        build()
        if not (os.path.exists(local) and os.path.getsize(local) > 1_000_000):
            raise SystemExit(f"[FAIL] Сборка {name} не дала результата.")
    sync_file(local, drive)
    return local


def train_env(extra):
    env = dict(os.environ)
    env.setdefault("FSTNET_EPOCHS", str(EPOCHS))
    env.setdefault("FSTNET_LR", LR)
    env.setdefault("FSTNET_CKPT_DIR", DRIVE_CKPT)
    env.update(extra)
    return env


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    logf = open(os.path.join(LOG_DIR, f"run_{ts}.log"), "w")

    log("=" * 60)
    log("JARVIS FULL TRAINING (Stage 1 + Stage 2) | 3B 1-bit MoF")
    log("=" * 60)
    mount_drive()
    ensure_repo()

    pip("tokenizers", "tqdm", "ijson")

    if args.fresh:
        for d in ("3b_mof", "3b_mof_stage2"):
            shutil.rmtree(os.path.join(SKILL, "checkpoints", d), ignore_errors=True)
            shutil.rmtree(os.path.join(DRIVE_CKPT, d), ignore_errors=True)
            log(f"--fresh: удалил {d} (локально и на Диске)")
        for c in ("jarvis_mof_samples.npz", "jarvis_mof_samples_stage2.npz"):
            os.path.exists(os.path.join(SKILL, "checkpoints", "3b_mof", c)) and \
                os.remove(os.path.join(SKILL, "checkpoints", "3b_mof", c))
        log("--fresh: локальные кэши очищены")

    if args.skip_data:
        log("--skip-data: датасеты не трогаю, обучение на том, что есть в data/")
    else:
        def build_stage1():
            log("Собираю jarvis_full: синтетика 200K + OpenHermes 300K + merge.")
            run(["python", "build_jarvis_data.py", "--count", "200000"],
                dict(os.environ), logf, timeout=2400)
            if run(["python", "fetch_openhermes.py", "--max-examples", "300000",
                    "--min-tokens", "50"], dict(os.environ), logf, timeout=3600) == 0 \
                    and os.path.exists(os.path.join(DATA, "jarvis_openhermes.json")):
                run(["python", "merge_datasets.py", "--weight", "0.7"],
                    dict(os.environ), logf)
            else:
                log("[WARN] OpenHermes не получен — Stage 1 на синтетике 200K.")

        def build_stage2():
            log("Собираю jarvis_special: синтетика 500K (Bash/tool-calling/Synth-Math).")
            run(["python", "build_specialized.py", "--count", "500000"],
                dict(os.environ), logf, timeout=1800)

        jarvis_full = ensure_data("jarvis_full.json",
                                  os.path.join(DRIVE_DATA, "jarvis_full.json"),
                                  build_stage1)
        jarvis_special = ensure_data("jarvis_special.json",
                                     os.path.join(DRIVE_DATA, "jarvis_special.json"),
                                     build_stage2)
        log(f"Stage 1 датасет: {jarvis_full}")
        log(f"Stage 2 датасет: {jarvis_special}")

    # ---------- Stage 1 ----------
    log("\n########## STAGE 1 (общая база, W0 float->1bit) ##########")
    rc1 = run(["python", "train_colab_mof.py"], train_env({"FSTNET_DATA": "data/jarvis_full.json"}), logf)
    if rc1 != 0:
        log("[FAIL] Stage 1 упал. Смотри лог выше. Stage 2 пропущен.")
        sync_file(os.path.join(LOG_DIR, f"run_{ts}.log"),
                  os.path.join(DRIVE_LOG, f"run_{ts}.log"))
        sys.exit(1)

    # Stage 2 требует moF_best.pt в checkpoints/3b_mof/ (локально + Диск)
    stage1_best = os.path.join(SKILL, "checkpoints", "3b_mof", "moF_best.pt")
    stage1_final = os.path.join(SKILL, "checkpoints", "3b_mof", "final.pt")
    if not os.path.exists(stage1_best) and os.path.exists(stage1_final):
        log("moF_best.pt нет (валидация не успела сохранить) — беру final.pt.")
        shutil.copyfile(stage1_final, stage1_best)
        sync_file(stage1_best, os.path.join(DRIVE_CKPT, "3b_mof", "moF_best.pt"))

    if args.stage1_only:
        log("--stage1-only: Stage 2 пропущен.")
    else:
        # ---------- Stage 2 ----------
        log("\n########## STAGE 2 (спец. датасет, W0 frozen, L_orth) ##########")
        rc2 = run(["python", "train_colab_mof.py"],
                  train_env({"FSTNET_STAGE": "2",
                             "FSTNET_DATA": "data/jarvis_special.json"}), logf)
        if rc2 != 0:
            log("[FAIL] Stage 2 упал. Смотри лог выше.")

    sync_file(os.path.join(LOG_DIR, f"run_{ts}.log"), os.path.join(DRIVE_LOG, f"run_{ts}.log"))
    log("\n" + "=" * 60)
    log("ГОТОВО. Артефакты:")
    log(f"  чекпоинты: {DRIVE_CKPT}/{{3b_mof,3b_mof_stage2}}/")
    log(f"  датасеты:  {DRIVE_DATA}/")
    log(f"  лог:       {DRIVE_LOG}/run_{ts}.log")
    log("Следующее: S3 — 1-bit export + bitnet.cpp (см. SPEC_3B_MOF.md).")
    log("=" * 60)


if __name__ == "__main__":
    main()
