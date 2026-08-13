#!/usr/bin/env python3
"""Раннер модели-учителя в Colab: Qwen 27B 1-bit (или любой GGUF) на llama.cpp server.

Учитель должен висеть на :8001 (OpenAI-совместимый /v1/chat/completions),
чтобы distill_colab.py мог генерировать траектории.

Откуда брать модель (приоритет env):
  TEACHER_GGUF_URL   — прямая ссылка на .gguf (скачивается в /content/models)
  TEACHER_HF_REPO    — репозиторий HF, TEACHER_HF_FILE — имя файла .gguf
  TEACHER_GGUF       — путь к уже скачанному .gguf локально

Запуск в Colab (T4 16GB, 1-bit Qwen ~3.5GB — помещается):
  !python run_teacher_colab.py \
      --hf-repo YOUR/REPO --hf-file model-1bit.gguf
  # или напрямую:
  !TEACHER_GGUF_URL=https://.../qwen-27b-1bit.gguf python run_teacher_colab.py

После готовности (лог "teacher ready") в той же сессии:
  !python distill_colab.py --count 200000 --workers 8
Env: TEACHER_GGUF_URL, TEACHER_HF_REPO, TEACHER_HF_FILE, TEACHER_GGUF,
     TEACHER_PORT (8001), TEACHER_N_GPU_LAYERS (99), TEACHER_CTX (4096)
"""
import os
import sys
import time
import argparse
import subprocess

def log(msg): print(msg, flush=True)

env = os.environ
PORT = int(env.get("TEACHER_PORT", "8001"))
N_GPU = int(env.get("TEACHER_N_GPU_LAYERS", "99"))
CTX = int(env.get("TEACHER_CTX", "4096"))
MODELS_DIR = "/content/models"

AP = argparse.ArgumentParser()
AP.add_argument("--gguf-url", default=env.get("TEACHER_GGUF_URL", ""), help="прямая ссылка на .gguf")
AP.add_argument("--hf-repo", default=env.get("TEACHER_HF_REPO", ""), help="HF репозиторий")
AP.add_argument("--hf-file", default=env.get("TEACHER_HF_FILE", ""), help="имя файла .gguf в репо")
AP.add_argument("--local-gguf", default=env.get("TEACHER_GGUF", ""), help="путь к локальному .gguf")
args = AP.parse_args()


def run(cmd, **kw):
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw)


def ensure_llama_server():
    log("Installing llama-cpp-python[server] (CUDA)...")
    r = subprocess.run(["pip", "install", "-q", "--force-reinstall", "--no-cache-dir",
                        "llama-cpp-python[server]>=0.3.6"])
    if r.returncode != 0:
        log("[WARN] CUDA build не удался — ставлю CPU-версию (медленнее, но рабочая)")
        subprocess.run(["pip", "install", "-q", "--force-reinstall", "--no-cache-dir",
                        "llama-cpp-python[server]>=0.3.6"])
    try:
        import llama_cpp.server  # noqa: F401
        return True
    except Exception as e:
        log(f"[FAIL] llama_cpp.server недоступен: {e}")
        return False


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if args.local_gguf:
        gguf = args.local_gguf
        log(f"Локальный GGUF: {gguf}")
        return gguf

    if args.gguf_url:
        fn = os.path.join(MODELS_DIR, os.path.basename(args.gguf_url))
        if os.path.exists(fn):
            log(f"Уже скачан: {fn}")
            return fn
        import urllib.request
        log(f"Download {args.gguf_url} -> {fn}")
        urllib.request.urlretrieve(args.gguf_url, fn)
        log(f"Готово: {os.path.getsize(fn)/1e9:.2f}GB")
        return fn

    if args.hf_repo and args.hf_file:
        from huggingface_hub import hf_hub_download
        fn = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file,
                             local_dir=MODELS_DIR, resume_download=True)
        log(f"HF download: {fn} ({os.path.getsize(fn)/1e9:.2f}GB)")
        return fn

    log("[FAIL] Укажи источник модели: --gguf-url | --hf-repo+--hf-file | --local-gguf")
    sys.exit(1)


def wait_ready(url, timeout=600):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    if not ensure_llama_server():
        sys.exit(1)
    gguf = download()
    url = f"http://127.0.0.1:{PORT}"
    log(f"Старт llama-server: {gguf} ctx={CTX} n_gpu_layers={N_GPU} port={PORT}")

    cmd = ["python", "-m", "llama_cpp.server",
           "--model", gguf,
           "--n_ctx", str(CTX),
           "--n_gpu_layers", str(N_GPU),
           "--host", "0.0.0.0",
           "--port", str(PORT),
           "--chat_format", "chatml"]
    log(f"{' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, universal_newlines=True, bufsize=1)

    if wait_ready(url):
        log("=" * 60)
        log(f"teacher ready: {url}/v1     модель загружена")
        log("Далее (в этой же сессии):")
        log("  !python distill_colab.py --count 200000 --workers 8")
        log("=" * 60)
        proc.wait()
    else:
        log("[FAIL] teacher не поднялся за 10 мин. Смотри лог:")
        import select
        for line in proc.stdout:
            log(line.rstrip())
        proc.kill()
        sys.exit(1)


if __name__ == "__main__":
    main()