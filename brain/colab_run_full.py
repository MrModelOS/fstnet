#!/usr/bin/env python3
"""Полный авто-запуск обучения JARVIS (Stage 1 + Stage 2) в Colab.

Одна ячейка (копируй целиком; работает и при повторном запуске):
  !wget -q https://raw.githubusercontent.com/MrModelOS/fstnet/master/colab_run_full.py
  %run colab_run_full.py

Важно: запускай через %run (не !python) — тогда монтирование Диска выполняется
в ядре ноутбука и покажет ссылку авторизации прямо в ячейке. Из !python
drive.mount() не работает (нет ядра). Если Диск не смонтирован — скрипт
продолжит локально, а после можно смонтировать ячейкой и перезапустить
(обучение продолжится с чекпоинта).

Что делает скрипт:
  1. Монтирует Google Диск (нужно разрешить в браузере; если нет — продолжает локально).
  2. Обеспечивает репозиторий в /content/fstnet: git clone, если нет; git pull, если есть.
  3. Ставит зависимости (tokenizers, tqdm, ijson; datasets — только для OpenHermes).
  4. Датасеты data/jarvis_full.json (Stage 1) и data/jarvis_special.json (Stage 2):
       локально есть -> берём; нет -> копируем с Диска MyDrive/fstnet/data/;
       нет нигде -> собираем (синтетика + OpenHermes для Stage 1, синтетика 500K для Stage 2);
       после обеспечения — синхронизируем на Диск (следующие запуски без сборки).
  5. Автопродолжение: обучение возобновляется с последнего чекпоинта (локально или на Диске);
     уже завершённая стадия (final.pt есть) пропускается:
       Stage 1:  FSTNET_EPOCHS=4 FSTNET_LR=2e-4 python jarvis_engine/trainer/run_trainer.py
                 -> checkpoints/3b_mof/moF_best.pt (локально и на Диске).
       Stage 2:  FSTNET_STAGE=2 FSTNET_DATA=data/jarvis_special.json python jarvis_engine/trainer/run_trainer.py
                 -> checkpoints/3b_mof_stage2/.
  6. Лог прогона -> Диск: MyDrive/fstnet/logs/run_<ts>.log.

Флаги:
  --stage1-only   только Stage 1
  --skip-data     не трогать датасеты (обучение на том, что уже в data/)
  --fresh         удалить чекпоинты (локально и на Диске) — чистый старт

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
DATA = os.path.join(SKILL, "brain", "data")
LOG_DIR = os.path.join(SKILL, "brain", "logs")
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
        log("Варианты:")
        log("  1. Запускай скрипт через %run (не !python) — тогда mount")
        log("     отработает и покажет ссылку авторизации прямо в ячейке.")
        log("  2. Смонтируй отдельной ячейкой:")
        log("       from google.colab import drive; drive.mount('/content/drive')")
        log("     и перезапусти скрипт — обучение продолжится с чекпоинта.")
        log("=" * 60)


def ensure_repo():
    if os.path.isdir(os.path.join(SKILL, ".git")):
        log("Репозиторий уже есть — git pull.")
        subprocess.run(["git", "-C", SKILL, "pull", "--ff-only", "-q"], check=False)
    else:
        if os.path.isdir(SKILL):
            stale = f"{SKILL}.stale.{int(time.time())}"
            os.rename(SKILL, stale)
            log(f"{SKILL} существует, но не git-репозиторий — убрал в {stale}")
        log(f"Клонирую репозиторий: {REPO}")
        subprocess.run(["git", "clone", "-q", REPO, SKILL], check=True)
    os.chdir(SKILL)
    sys.path.insert(0, os.path.join(SKILL, "brain"))
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
    if drive_mounted():
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
            shutil.rmtree(os.path.join(SKILL, "brain", "checkpoints", d), ignore_errors=True)
            shutil.rmtree(os.path.join(DRIVE_CKPT, d), ignore_errors=True)
            log(f"--fresh: удалил {d} (локально и на Диске)")
        for c in ("jarvis_mof_samples.npz", "jarvis_mof_samples_stage2.npz"):
            os.path.exists(os.path.join(SKILL, "brain", "checkpoints", "3b_mof", c)) and \
                os.remove(os.path.join(SKILL, "brain", "checkpoints", "3b_mof", c))
        log("--fresh: локальные кэши очищены")

    if args.skip_data:
        log("--skip-data: датасеты не трогаю, обучение на том, что есть в data/")
    else:
        def build_stage1():
            log("Собираю jarvis_full: синтетика 200K + OpenHermes 300K + merge.")
            run(["python", "brain/build_jarvis_data.py", "--count", "200000"],
                dict(os.environ), logf, timeout=2400)
            if run(["python", "brain/fetch_openhermes.py", "--max-examples", "300000",
                    "--min-tokens", "50"], dict(os.environ), logf, timeout=3600) == 0 \
                    and os.path.exists(os.path.join(DATA, "jarvis_openhermes.json")):
                run(["python", "brain/merge_datasets.py", "--weight", "0.7"],
                    dict(os.environ), logf)
            else:
                log("[WARN] OpenHermes не получен — Stage 1 на синтетике 200K.")

        def build_stage2():
            log("Собираю jarvis_special: синтетика 500K (Bash/tool-calling/Synth-Math).")
            run(["python", "brain/build_specialized.py", "--count", "500000"],
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
    def has_ckpt(subdir, name):
        return (os.path.exists(os.path.join(SKILL, "brain", "checkpoints", subdir, name))
                or (drive_mounted()
                    and os.path.exists(os.path.join(DRIVE_CKPT, subdir, name))))

    def ensure_stage1_best():
        """Stage 2 требует moF_best.pt в checkpoints/3b_mof/ (локально + Диск)."""
        best = os.path.join(SKILL, "brain", "checkpoints", "3b_mof", "moF_best.pt")
        if not os.path.exists(best):
            os.makedirs(os.path.dirname(best), exist_ok=True)
            for cand in (os.path.join(SKILL, "brain", "checkpoints", "3b_mof", "final.pt"),
                         os.path.join(DRIVE_CKPT, "3b_mof", "final.pt")):
                if os.path.exists(cand):
                    shutil.copyfile(cand, best)
                    log("moF_best.pt нет (валидация не успела сохранить) — беру final.pt.")
                    break
        sync_file(best, os.path.join(DRIVE_CKPT, "3b_mof", "moF_best.pt"))

    if has_ckpt("3b_mof", "final.pt"):
        log("Stage 1 уже завершён (final.pt есть) — пропускаю обучение Stage 1.")
    else:
        log("\n########## STAGE 1 (общая база, W0 float->1bit) ##########")
        rc1 = run(["python", "jarvis_engine/trainer/run_trainer.py"], train_env({"FSTNET_DATA": "brain/data/jarvis_full.json"}), logf)
        if rc1 != 0:
            log("[FAIL] Stage 1 упал. Смотри лог выше. Stage 2 пропущен.")
            sync_file(os.path.join(LOG_DIR, f"run_{ts}.log"),
                      os.path.join(DRIVE_LOG, f"run_{ts}.log"))
            sys.exit(1)
        ensure_stage1_best()

    if args.stage1_only:
        log("--stage1-only: Stage 2 пропущен.")
    elif has_ckpt("3b_mof_stage2", "final.pt"):
        log("Stage 2 уже завершён (final.pt есть) — пропускаю обучение Stage 2.")
    else:
        # ---------- Stage 2 ----------
        if not (has_ckpt("3b_mof", "moF_best.pt") or has_ckpt("3b_mof", "final.pt")):
            log("[FAIL] Нет чекпоинта Stage 1 (3b_mof/moF_best.pt) — Stage 2 невозможен.")
        else:
            ensure_stage1_best()
            log("\n########## STAGE 2 (спец. датасет, W0 frozen, L_orth) ##########")
            rc2 = run(["python", "jarvis_engine/trainer/run_trainer.py"],
                      train_env({"FSTNET_STAGE": "2",
                                 "FSTNET_DATA": "brain/data/jarvis_special.json"}), logf)
            if rc2 != 0:
                log("[FAIL] Stage 2 упал. Смотри лог выше.")

    sync_file(os.path.join(LOG_DIR, f"run_{ts}.log"), os.path.join(DRIVE_LOG, f"run_{ts}.log"))
    log("\n" + "=" * 60)
    log("ГОТОВО. Артефакты:")
    log(f"  чекпоинты: {DRIVE_CKPT}/{{3b_mof,3b_mof_stage2}}/")
    log(f"  датасеты:  {DRIVE_DATA}/")
    log(f"  лог:       {DRIVE_LOG}/run_{ts}.log")
    log("Следующее: S3 — 1-bit export + bitnet.cpp (см. SPEC_3B_MOF.md).")
    if not drive_mounted():
        log("")
        log("ВНИМАНИЕ: Диск не был смонтирован — артефакты остались только в /content.")
        log("Смонтируй ячейкой: from google.colab import drive; drive.mount('/content/drive')")
        log("и перезапусти скрипт — он докачает чекпоинты на Диск (resume).")
    log("=" * 60)


if __name__ == "__main__":
    main()
