#!/usr/bin/env python3
"""Дистилляция датасета JARVIS с учителя Qwen 3.6 27B Bonsai (1-bit).

Учитель: vLLM / llama.cpp server с OpenAI-совместимым API (chat/completions).
Fallback: если сервер недоступен — локальная синтетика из build_jarvis_data.

Выход:
  data/distill.jsonl   — прогрессивно дописываемые траектории
  data/jarvis_full.json — финальный массив [[role, content], ...] для train_colab_mof.py

Запуск:
  # терминал учителя (Colab T4/Kaggle P100):
  #   llama-server -m qwen27b-1bit.gguf -c 2048 --port 8001 --api-key none
  %cd fstnet
  !python distill_colab.py --count 200000 --workers 8 --base-url http://localhost:8001/v1
  !python distill_colab.py --to-json --synthetic 40000   # собрать финальный датасет
Env: TEACHER_URL, TEACHER_MODEL, TEACHER_KEY, JARVIS_SEED
"""
import os
import sys
import json
import time
import hashlib
import random
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

def log(msg): print(msg, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_URL = os.environ.get("TEACHER_URL", "http://localhost:8001/v1")
MODEL = os.environ.get("TEACHER_MODEL", "qwen3.6-27b-bonsai-1bit")
KEY = os.environ.get("TEACHER_KEY", "none")
SEED = int(os.environ.get("JARVIS_SEED", "42"))

from build_jarvis_data import (SYSTEM_PROMPT, CODE_SNIPPETS, MATH_QS,
                               REASON_QS, TOOL_REQUESTS, SMALL_CHAT)

TEACHER_SYSTEM = (
    SYSTEM_PROMPT
    + " Reply ONLY in this exact structure: <think>your reasoning</think> "
      "then <tool_call>{\"name\": \"...\", \"args\": {...}}</tool_call> if a tool is needed, "
      "then your final answer to the user addressing them as 'Sir'."
)

KINDS = ["code", "reason", "tool", "chat"]
FRAC = {"code": 0.40, "reason": 0.25, "tool": 0.20, "chat": 0.15}


def build_user(kind, rng):
    if kind == "code":
        fn = rng.choice(["quicksort", "binary_search", "parse_config", "download_file",
                         "merge_dicts", "benchmark", "file_watcher", "log_rotate",
                         "retry_call", "read_csv", "cache_get", "init_workspace"])
        task = rng.choice(["sort a list in place", "search a sorted array", "parse a config",
                           "download and save a URL", "merge two dicts recursively",
                           "run a micro-benchmark", "watch a directory for changes",
                           "rotate log files", "retry a callable N times",
                           "read a CSV into dicts", "manage a TTL cache",
                           "initialize a workspace layout"])
        tpl = rng.choice([
            f"Write a Python function {fn} that {task}.",
            f"Refactor this function to be readable and efficient:\n{rng.choice(CODE_SNIPPETS)}",
            f"Write a bash script that {task}.",
            f"Write a unit test for {fn}.",
            f"Explain this code:\n{rng.choice(CODE_SNIPPETS)}"])
        return tpl
    if kind == "reason":
        q, ans, steps = rng.choice(MATH_QS + REASON_QS)
        return q
    if kind == "tool":
        req, fn, _, _ = rng.choice(TOOL_REQUESTS)
        return f"Sir, {req}."
    q, a = rng.choice(SMALL_CHAT)
    return q


def pick_kind(rng):
    r = rng.random()
    if r < FRAC["code"]: return "code"
    if r < FRAC["code"] + FRAC["reason"]: return "reason"
    if r < FRAC["code"] + FRAC["reason"] + FRAC["tool"]: return "tool"
    return "chat"


def content_hash(kind, user):
    return hashlib.sha1(f"{kind}|{user}".encode()).hexdigest()


def validate(kind, text):
    text = (text or "").strip()
    if len(text) < 12 or len(text) > 4000:
        return False, "len"
    if kind == "tool" and "<tool_call>" not in text:
        return False, "no_tool_call"
    return True, "ok"


class Distiller:
    def __init__(self, base_url=BASE_URL, model=MODEL, key=KEY, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {key}"} if key and key != "none" else {}
        self.available = self._ping()

    def _ping(self):
        try:
            import requests
            r = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=10)
            return r.ok
        except Exception:
            return False

    def chat(self, user):
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.95,
            "top_k": 20,
        }
        r = requests.post(f"{self.base_url}/chat/completions",
                          json=payload, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def synthetic(kind, rng):
    if kind == "tool":
        req, fn, args, answer = rng.choice(TOOL_REQUESTS)
        call = f'<tool_call>{{"name": "{fn}", "args": {json.dumps(args)}}}</tool_call>'
        conv = [("system", SYSTEM_PROMPT), ("user", f"Sir, {req}."),
                ("assistant", call), ("system", '{"ok": true}'), ("assistant", answer)]
    else:
        from build_jarvis_data import code_sample, reasoning_sample, chat_sample
        builder = {"code": code_sample, "reason": reasoning_sample, "chat": chat_sample}[kind]
        sysp, q, inp, ans = builder(rng)
        conv = [("system", sysp), ("user", f"{q}\n{inp}" if inp else q), ("assistant", ans)]
    return conv


def produce(kind, rng, distiller):
    user = build_user(kind, rng)
    cid = content_hash(kind, user)
    if distiller is not None and distiller.available:
        try:
            ans = distiller.chat(user)
            ok, why = validate(kind, ans)
            if ok:
                return cid, kind, [("system", SYSTEM_PROMPT), ("user", user), ("assistant", ans)]
            return cid, kind, None
        except Exception as e:
            log(f"[warn] teacher error: {e} — fallback synthetic")
    conv = synthetic(kind, rng)
    return cid, kind, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=int(os.environ.get("JARVIS_COUNT", "200000")))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="data/distill.jsonl")
    ap.add_argument("--to-json", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--synthetic-out", default="data/jarvis_full.json")
    args = ap.parse_args()

    if args.to_json:
        seen, convs = set(), []
        if os.path.exists(args.out):
            for line in open(args.out):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cid = obj.get("cid")
                if cid in seen:
                    continue
                seen.add(cid)
                convs.append(obj["conv"])
        if args.synthetic:
            log(f"Добавляю {args.synthetic} синтетических...")
            rng = random.Random(SEED)
            for _ in range(args.synthetic):
                convs.append(synthetic(pick_kind(rng), rng))
        os.makedirs(os.path.dirname(args.synthetic_out) or ".", exist_ok=True)
        with open(args.synthetic_out, "w") as f:
            json.dump(convs, f)
        log(f"FINAL -> {args.synthetic_out}: {len(convs)} конверсаций")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                seen.add(json.loads(line)["cid"])
            except Exception:
                continue
        log(f"Resume: {len(seen)} уже есть в {args.out}")

    distiller = Distiller(args.base_url, args.model)
    log(f"Teacher {args.base_url}: {'ONLINE' if distiller.available else 'OFFLINE (synthetic fallback)'}")

    rng = random.Random(SEED)
    done = len(seen)
    target = max(args.count - done, 0)
    stats = {k: 0 for k in KINDS}
    rejects = 0
    t0 = time.time()

    def task(_):
        kind = pick_kind(rng)
        return produce(kind, rng, distiller)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task, i) for i in range(max(target, args.workers))]
        with open(args.out, "a") as f:
            while done < args.count:
                for fu in futs:
                    if not fu.done():
                        continue
                    cid, kind, conv = fu.result()
                    futs.remove(fu)
                    futs.append(ex.submit(task, 0))
                    if conv is not None and cid not in seen:
                        seen.add(cid)
                        f.write(json.dumps({"cid": cid, "kind": kind, "conv": conv}) + "\n")
                        f.flush()
                        stats[kind] += 1
                        done += 1
                        if done % 50 == 0:
                            el = time.time() - t0
                            rate = done / max(el, 1)
                            eta = (args.count - done) / max(rate, 1e-9)
                            log(f"{done}/{args.count} | {rate:.1f}/s | ETA {eta/60:.1f}min")
                    elif conv is None:
                        rejects += 1

    el = time.time() - t0
    dist = {k: f"{v} ({v/max(done,1)*100:.0f}%)" for k, v in stats.items()}
    log(f"DONE: {done} траекторий -> {args.out} | rejects={rejects} | {el/60:.1f}min | {dist}")
    log(f"Дальше: !python distill_colab.py --to-json --synthetic 40000")


if __name__ == "__main__":
    main()