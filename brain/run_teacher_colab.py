#!/usr/bin/env python3
"""Раннер модели-учителя в Colab: Bonsai-27B 1-bit (Qwen3.6-27B) на форке PrismML llama.cpp.

Модель: prism-ml/Bonsai-27B-gguf — `Bonsai-27B-Q1_0.gguf` (3.8GB, 1.125 bit/weight).
ВАЖНО: кастомный формат Q1_0_g128 + гибридное внимание -> нужен ФОРК llama.cpp
       PrismML (https://github.com/PrismML-Eng/llama.cpp), НЕ стандартный llama.cpp
       и НЕ pip llama-cpp-python. Скрипт собирает форк из исходников с CUDA.

Учитель виснет на :8001 (OpenAI-совместимый /v1/chat/completions).

Запуск в Colab (T4 16GB, сборка ~5-10 мин, модель 3.8GB):
  !python run_teacher_colab.py
  # после "teacher ready" — той же сессией:
  !python distill_colab.py --count 200000 --workers 8

Замена модели (env): TEACHER_HF_REPO / TEACHER_HF_FILE / TEACHER_GGUF_URL / TEACHER_GGUF
Прочее: TEACHER_PORT (8001), TEACHER_CTX (8192), TEACHER_N_GPU_LAYERS (99)
"""
import os
import sys
import time
import shutil
import argparse
import subprocess

def log(msg): print(msg, flush=True)

env = os.environ
PORT = int(env.get("TEACHER_PORT", "8001"))
N_GPU = int(env.get("TEACHER_N_GPU_LAYERS", "99"))
CTX = int(env.get("TEACHER_CTX", "8192"))
MODELS_DIR = "/content/models"
LLAMA_DIR = "/content/llama.cpp"
FORK_URL = "https://github.com/PrismML-Eng/llama.cpp"
DEFAULT_REPO = "prism-ml/Bonsai-27B-gguf"
DEFAULT_FILE = "Bonsai-27B-Q1_0.gguf"

AP = argparse.ArgumentParser()
AP.add_argument("--gguf-url", default=env.get("TEACHER_GGUF_URL", ""), help="прямая ссылка .gguf (не формат Q1_0_g128 рекомендован)")
AP.add_argument("--hf-repo", default=env.get("TEACHER_HF_REPO", DEFAULT_REPO))
AP.add_argument("--hf-file", default=env.get("TEACHER_HF_FILE", DEFAULT_FILE))
AP.add_argument("--local-gguf", default=env.get("TEACHER_GGUF", ""))
AP.add_argument("--no-build", action="store_true", help="использовать готовый build (без компиляции)")
AP.add_argument("--jobs", type=int, default=max(2, os.cpu_count() or 2))
args = AP.parse_args()


def run(cmd, **kw):
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, **kw)


def ensure_llama_server():
    bin_path = os.path.join(LLAMA_DIR, "build", "bin", "llama-server")
    if os.path.exists(bin_path):
        log(f"llama-server уже собран: {bin_path}")
        return bin_path, False

    if not args.no_build:
        log("Сборка форка PrismML llama.cpp с CUDA (5-10 мин)...")
        if not os.path.isdir(os.path.join(LLAMA_DIR, ".git")):
            r = run(["git", "clone", "--depth", "1", FORK_URL, LLAMA_DIR])
            if r.returncode != 0:
                log("[FAIL] не удалось клонировать форк PrismML")
                return None, False
        for r in (run(["cmake", "-B", os.path.join(LLAMA_DIR, "build"),
                       "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release", LLAMA_DIR]),
                  run(["cmake", "--build", os.path.join(LLAMA_DIR, "build"), "-j",
                       str(args.jobs), "--target", "llama-server"])):
            if r.returncode != 0:
                log("[FAIL] сборка llama.cpp упала")
                return None, False
        if os.path.exists(bin_path):
            return bin_path, True

    log("[FAIL] build/bin/llama-server отсутствует. Отключи --no-build или собери вручную.")
    return None, False


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if args.local_gguf:
        log(f"Локальный GGUF: {args.local_gguf}")
        return os.path.abspath(args.local_gguf)
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
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=True)
            from huggingface_hub import hf_hub_download
        fn = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file,
                             local_dir=MODELS_DIR, resume_download=True)
        log(f"HF download: {fn} ({os.path.getsize(fn)/1e9:.2f}GB)")
        return fn
    log("[FAIL] укажи --gguf-url | --hf-repo+--hf-file | --local-gguf")
    sys.exit(1)


def wait_ready(url, timeout=900):
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
    bin_path, _ = ensure_llama_server()
    if not bin_path:
        sys.exit(1)
    gguf = download()
    url = f"http://127.0.0.1:{PORT}"

    cmd = [bin_path, "-m", gguf,
           "--host", "0.0.0.0", "--port", str(PORT),
           "-ngl", str(N_GPU),
           "-c", str(CTX),
           "--flash-attn", "on",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--temp", "0.7", "--top-p", "0.95", "--top-k", "20"]
    log(f"Старт llama-server: {gguf} ctx={CTX} ngl={N_GPU} port={PORT}")
    log("  " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    if wait_ready(url):
        log("=" * 60)
        log(f"teacher ready: {url}/v1   модель: {args.hf_repo}/{args.hf_file}")
        log("Далее (той же сессией):")
        log("  !python distill_colab.py --count 200000 --workers 8")
        log("=" * 60)
        proc.wait()
    else:
        log("[FAIL] teacher не поднялся за 15 мин. Хвост лога:")
        import select
        for line in proc.stdout:
            log(line.rstrip())
        proc.kill()
        sys.exit(1)


if __name__ == "__main__":
    main()