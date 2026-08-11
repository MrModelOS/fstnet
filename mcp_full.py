#!/usr/bin/env python3
"""Full MCP Server: STDM + ACSC + GGM + Session Isolation."""
import os, json, time, sqlite3, hashlib, threading, requests
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── STDM ────────────────────────────────────────────────
import ast
from pathlib import Path

STDM_DB = "data/ggm/stdm.db"
os.makedirs("data/ggm", exist_ok=True)

def get_stdm():
    db = sqlite3.connect(STDM_DB, check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, file TEXT, name TEXT, kind TEXT, signature TEXT, body_hash TEXT, body TEXT, updated_at REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT, kind TEXT, PRIMARY KEY(src,dst))")
    db.commit()
    return db

def index_file(filepath):
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return 0
    db = get_stdm()
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            sig = f"def {node.name}({', '.join(args)})"
            body = ast.get_source_segment(source, node) or ""
            nid = hashlib.md5(f"{filepath}:{node.name}".encode()).hexdigest()[:16]
            db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                (nid, filepath, node.name, "func", sig, hashlib.md5(body.encode()).hexdigest(), body[:500], "", time.time(), time.time()))
            count += 1
        elif isinstance(node, ast.ClassDef):
            sig = f"class {node.name}"
            body = ast.get_source_segment(source, node) or ""
            nid = hashlib.md5(f"{filepath}:{node.name}:class".encode()).hexdigest()[:16]
            db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                (nid, filepath, node.name, "class", sig, hashlib.md5(body.encode()).hexdigest(), body[:500], "", time.time(), time.time()))
            count += 1
    db.commit()
    return count

# ── Sessions ─────────────────────────────────────────────
sessions = {}
session_lock = threading.Lock()

def get_session(chat_id):
    with session_lock:
        if chat_id not in sessions:
            db = sqlite3.connect(f"data/ggm/session_{chat_id}.db", check_same_thread=False)
            db.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, concept TEXT, content TEXT, created_at REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT)")
            db.commit()
            sessions[chat_id] = {"db": db, "last_access": time.time()}
        else:
            sessions[chat_id]["last_access"] = time.time()
        return sessions[chat_id]

# ── Handler ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

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
        if self.path == "/status":
            self._json(200, {"ok": True, "sessions": list(sessions.keys())})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/mcp":
            self._json(404, {"error": "use /mcp"})
            return
        data = self._read()
        tool = data.get("tool", "")
        chat_id = data.get("chat_id", "default")

        try:
            if tool == "ggm_search":
                query = data.get("query", "")
                sess = get_session(chat_id)
                rows = sess["db"].execute("SELECT concept, content FROM nodes WHERE concept LIKE ? OR content LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", data.get("top_k", 3))).fetchall()
                results = [{"concept": r[0], "content": r[1]} for r in rows]
                # Also search STDM
                stdm = get_stdm()
                rows2 = stdm.execute("SELECT name, signature FROM nodes WHERE name LIKE ? OR signature LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", 3)).fetchall()
                for r in rows2:
                    results.append({"concept": r[0], "content": r[1], "source": "stdm"})
                self._json(200, {"tool": tool, "results": results})

            elif tool == "ggm_insert":
                sess = get_session(chat_id)
                nid = hashlib.md5(f"{data.get('concept','')}:{data.get('content','')[:50]}".encode()).hexdigest()[:12]
                sess["db"].execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?)",
                    (nid, data.get("concept", ""), data.get("content", ""), time.time()))
                sess["db"].commit()
                self._json(200, {"tool": tool, "node_id": nid})

            elif tool == "stdm_index":
                path = data.get("path", ".")
                count = 0
                if os.path.isfile(path):
                    count = index_file(path)
                else:
                    for dp, _, files in os.walk(path):
                        for f in files:
                            if f.endswith(".py"):
                                count += index_file(os.path.join(dp, f))
                self._json(200, {"tool": tool, "indexed": count})

            elif tool == "stdm_context":
                query = data.get("query", "")
                stdm = get_stdm()
                rows = stdm.execute("SELECT name, signature FROM nodes WHERE name LIKE ? OR signature LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", 5)).fetchall()
                self._json(200, {"tool": tool, "results": [{"name": r[0], "signature": r[1]} for r in rows]})

            elif tool == "ggm_prune_session":
                sess = get_session(chat_id)
                cutoff = time.time() - 3600
                sess["db"].execute("DELETE FROM nodes WHERE created_at < ?", (cutoff,))
                sess["db"].commit()
                self._json(200, {"tool": tool, "pruned": True})

            elif tool == "chat":
                query = data.get("user_message", "")
                sess = get_session(chat_id)
                # Get context from session + STDM
                ctx_rows = sess["db"].execute("SELECT concept, content FROM nodes WHERE concept LIKE ? OR content LIKE ? LIMIT 2",
                    (f"%{query}%", f"%{query}%")).fetchall()
                context = "\n".join([f"### {r[0]}\n{r[1]}" for r in ctx_rows])
                self._json(200, {"tool": tool, "context": context})

            else:
                self._json(400, {"error": f"unknown: {tool}"})
        except Exception as e:
            self._json(500, {"error": str(e)})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"MCP full server on :{args.port}", flush=True)
    httpd.serve_forever()
