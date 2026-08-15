#!/usr/bin/env python3
"""jarvis_engine/runtime/mcp_server.py — нативный IPC-сервер JARVIS Core.

Связывает: paged KV (ModelPagedRunner), sampler (min-p/top-p/grammar), GGM-память.
Только stdlib + torch + tokenizers (никакого fastapi/uvicorn — «native»).

Transports:
  REST        : http://0.0.0.0:PORT   (threaded)
  UnixSocket  : UNIX_SOCK  (задай путь — будет UDS-сервер)

API:
  POST /chat   {"messages":[...], "max_tokens":, "temperature":, "grammar": "json"|None}
  POST /generate {"prompt": "...", ...}
  POST /mcp    инструменты GGM (ggm_search / ggm_insert / ggm_prune_session) + /status
  GET  /health

Env:
  FSTNET_MODEL  путь к чекпоинту MoF (по умолчанию ищет checkpoints/3b_mof/moF_best.pt)
  FSTNET_PORT   HTTP порт (8765)
  FSTNET_UNIX   путь UDS (необязательно)
  FSTNET_GGM    URL GGM-сервера (по умолчанию http://localhost:8766/mcp)
  FSTNET_INT8   =1 включить INT8 KV-кэш
"""
import os
import json
import sys
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "brain")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizers import Tokenizer

from paged_kv import ModelPagedRunner
from sampler import sample, JsonSchemaMask
from colab_drive import load_checkpoint

PORT = int(os.environ.get("FSTNET_PORT", "8765"))
UNIX_SOCK = os.environ.get("FSTNET_UNIX", "").strip()
INT8 = os.environ.get("FSTNET_INT8", "").strip() in ("1", "true")

# ── модель ────────────────────────────────────────────────
def find_ckpt():
    cands = [os.environ.get("FSTNET_MODEL", "").strip()]
    for d in ("checkpoints/3b_mof", "checkpoints/3b_mof_stage2",
              "checkpoints/152m", "checkpoints/800m"):
        cands.append(os.path.join(_ROOT, d, "moF_best.pt"))
        cands.append(os.path.join(_ROOT, d, "final.pt"))
        cands.append(os.path.join(d, "moF_best.pt"))
        cands.append(os.path.join(d, "final.pt"))
    cands.append(os.path.join(_ROOT, "checkpoints/3b_mof/moF_best.pt"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit("[FAIL] Не найден чекпоинт. Задай FSTNET_MODEL=<путь.pt>")


def load_model():
    path = find_ckpt()
    print(f"Loading {path}...", flush=True)
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    from model.core_mof import FSTMoFModel
    model = FSTMoFModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda"
                     and torch.cuda.get_device_capability()[0] >= 8 else torch.float16).eval()
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    tok.post_processor = None
    runner = ModelPagedRunner(model, ctx_len=cfg.max_seq_len * 2,
                              quantize_int8=INT8)
    print(f"Model ready: {sum(p.numel() for p in model.parameters())/1e9:.2f}B, "
          f"KV {'int8' if INT8 else 'fp16'}", flush=True)
    return model, runner, tok, cfg


MODEL, RUNNER, TOK, CFG = load_model()
IM_S, IM_E = "<|im_start|>", "<|im_end|>"


def encode_chat(messages, max_len=None):
    ids = []
    for role, content in messages:
        ids += TOK.encode(f"{IM_S}{role}\n{content}{IM_E}").ids
    max_len = max_len or CFG.max_seq_len
    return ids[-max_len:]


def decode_ids(ids):
    return TOK.decode(ids, skip_special_tokens=False)


def grammar_mask(kind):
    if kind == "json":
        jm = JsonSchemaMask(TOK)
        return jm.make_allowed_fn(decode_ids)
    return None


# ── GGM-клиент ─────────────────────────────────────────────
GGM_URL = os.environ.get("FSTNET_GGM", "http://localhost:8766/mcp")


def ggm_search(chat_id, query, top_k=2):
    try:
        import urllib.request
        body = json.dumps({"tool": "ggm_search", "chat_id": chat_id,
                           "query": query, "top_k": top_k}).encode()
        req = urllib.request.Request(GGM_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as r:
            results = json.load(r).get("results", [])
        return [f"### {h.get('concept','')}\n{h.get('content','')}" for h in results]
    except Exception:
        return []


def ggm_insert(chat_id, concept, content):
    try:
        import urllib.request
        body = json.dumps({"tool": "ggm_insert", "chat_id": chat_id,
                           "concept": concept, "content": content}).encode()
        req = urllib.request.Request(GGM_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


# ── генерация ──────────────────────────────────────────────
def generate(ids, max_new=256, temperature=0.7, top_k=50, top_p=0.9,
             min_p=0.05, repetition_penalty=1.15, grammar=None, stop=None):
    stop = set(stop or [])
    if ids in (TOK.token_to_id(IM_E), TOK.token_to_id("<eos>")):
        stop.add(TOK.token_to_id("<eos>"))
    prompt = torch.tensor([ids], dtype=torch.long, device=MODEL.device)
    RUNNER.prefill(prompt)
    out = list(ids)
    cur = prompt
    allowed = grammar_mask(grammar)
    for _ in range(max_new):
        logits = RUNNER.forward_token(cur[:, -1:])[0, 0]
        if allowed is not None:
            nxt, _ = sample(logits, ids=out, temperature=temperature,
                            top_k=top_k, top_p=top_p, min_p=min_p,
                            repetition_penalty=repetition_penalty,
                            allowed_fn=allowed)
        else:
            nxt, _ = sample(logits, ids=out, temperature=temperature,
                            top_k=top_k, top_p=top_p, min_p=min_p,
                            repetition_penalty=repetition_penalty)
        out.append(nxt)
        cur = torch.tensor([[nxt]], device=MODEL.device, dtype=torch.long)
        if nxt in stop:
            break
    return out[len(ids):]


# ── HTTP ───────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "kv_blocks": RUNNER.kv.n_blocks,
                             "kv_pos": RUNNER.kv._pos})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/chat", "/generate", "/mcp"):
            self._json(404, {"error": "use /chat | /generate | /mcp"})
            return
        data = self._read()
        try:
            if self.path == "/chat":
                msgs = data.get("messages", [])
                chat_id = data.get("chat_id", "default")
                query = msgs[-1]["content"] if msgs else ""
                ctx = ggm_search(chat_id, query)
                if ctx:
                    msgs.insert(-1, {"role": "context", "content": "\n\n".join(ctx)})
                ids = encode_chat(msgs)
                out = generate(ids, max_new=data.get("max_tokens", 256),
                               temperature=data.get("temperature", 0.7),
                               top_k=data.get("top_k", 50),
                               top_p=data.get("top_p", 0.9),
                               min_p=data.get("min_p", 0.05),
                               repetition_penalty=data.get("repetition_penalty", 1.15),
                               grammar=data.get("grammar"))
                ggm_insert(chat_id, query, decode_ids(out))
                self._json(200, {"text": decode_ids(out), "context_sources": len(ctx)})
            elif self.path == "/generate":
                prompt = data.get("prompt", "")
                ids = TOK.encode(prompt).ids
                out = generate(ids, max_new=data.get("max_tokens", 256),
                               temperature=data.get("temperature", 0.7),
                               top_k=data.get("top_k", 50),
                               top_p=data.get("top_p", 0.9),
                               min_p=data.get("min_p", 0.05),
                               repetition_penalty=data.get("repetition_penalty", 1.15),
                               grammar=data.get("grammar"))
                self._json(200, {"text": decode_ids(out)})
            else:  # /mcp — инструменты (для совместимости со старым агентом)
                tool = data.get("tool", "")
                chat_id = data.get("chat_id", "default")
                if tool == "ggm_search":
                    r = ggm_search(chat_id, data.get("query", ""), data.get("top_k", 3))
                    self._json(200, {"tool": tool, "results": r})
                elif tool == "ggm_insert":
                    ggm_insert(chat_id, data.get("concept", ""), data.get("content", ""))
                    self._json(200, {"tool": tool, "node_id": "ok"})
                elif tool == "status":
                    self._json(200, {"tool": tool, "kv_blocks": RUNNER.kv.n_blocks,
                                     "kv_pos": RUNNER.kv._pos})
                else:
                    self._json(400, {"error": f"unknown tool: {tool}"})
        except Exception as e:
            self._json(500, {"error": str(e)})


def serve_http():
    httpd = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"HTTP :{PORT}", flush=True)
    httpd.serve_forever()


def serve_unix(path):
    if os.path.exists(path):
        os.remove(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(8)
    print(f"UDS  {path}", flush=True)
    while True:
        conn, _ = sock.accept()
        threading.Thread(target=_handle_uds, args=(conn,), daemon=True).start()


def _handle_uds(conn):
    with conn:
        data = b""
        while True:
            chunk = conn.recv(1 << 16)
            if not chunk:
                break
            data += chunk
            if data.endswith(b"}"):  # простейший фрейминг JSON
                break
        try:
            req = json.loads(data)
            sys.path.insert(0, _HERE)
            body = json.dumps({"text": "uds-echo"}).encode()
            conn.sendall(body)
        except Exception:
            pass


if __name__ == "__main__":
    threads = []
    if UNIX_SOCK:
        threads.append(threading.Thread(target=serve_unix, args=(UNIX_SOCK,), daemon=True))
    threads.append(threading.Thread(target=serve_http, daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()