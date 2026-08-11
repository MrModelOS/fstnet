#!/usr/bin/env python3
"""MCP Server - session-isolated GGM dispatcher."""
import os, json, time, sqlite3, hashlib, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

GGG_DIR = "data/ggm"
os.makedirs(GGG_DIR, exist_ok=True)


class SessionManager:
    def __init__(self, max_sessions=10):
        self.max_sessions = max_sessions
        self.sessions = {}
        self.global_db = self._init_db("global")
        self.lock = threading.Lock()

    def _init_db(self, name):
        path = os.path.join(GGG_DIR, f"ggm_{name}.db")
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, content TEXT, concept TEXT, meta TEXT, created_at REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT, weight REAL, PRIMARY KEY(src,dst))")
        conn.commit()
        return conn

    def get_session(self, chat_id):
        with self.lock:
            if chat_id not in self.sessions:
                if len(self.sessions) >= self.max_sessions:
                    oldest = min(self.sessions, key=lambda k: self.sessions[k]["last_access"])
                    self._close_session(oldest)
                db = self._init_db(f"session_{chat_id}")
                self.sessions[chat_id] = {"db": db, "last_access": time.time(), "nodes": 0}
            else:
                self.sessions[chat_id]["last_access"] = time.time()
            return self.sessions[chat_id]

    def _close_session(self, chat_id):
        if chat_id in self.sessions:
            try:
                self.sessions[chat_id]["db"].close()
            except Exception:
                pass
            del self.sessions[chat_id]

    def search(self, chat_id, query, top_k=3):
        results = []
        rows = self.global_db.execute("SELECT id, content, concept FROM nodes WHERE content LIKE ? OR concept LIKE ? LIMIT ?", (f"%{query}%", f"%{query}%", top_k)).fetchall()
        for r in rows:
            results.append({"id": r[0], "content": r[1], "concept": r[2], "source": "global"})
        sess = self.get_session(chat_id)
        rows = sess["db"].execute("SELECT id, content, concept FROM nodes WHERE content LIKE ? OR concept LIKE ? LIMIT ?", (f"%{query}%", f"%{query}%", top_k)).fetchall()
        for r in rows:
            results.append({"id": r[0], "content": r[1], "concept": r[2], "source": "session"})
        return results[:top_k]

    def insert(self, chat_id, concept, content, meta=None):
        sess = self.get_session(chat_id)
        nid = hashlib.md5(f"{concept}:{content[:50]}".encode()).hexdigest()[:12]
        sess["db"].execute("INSERT OR REPLACE INTO nodes (id, content, concept, meta, created_at) VALUES (?,?,?,?,?)", (nid, content, concept, json.dumps(meta or {}), time.time()))
        sess["db"].commit()
        sess["nodes"] += 1
        return nid

    def prune(self, chat_id):
        sess = self.get_session(chat_id)
        cutoff = time.time() - 3600
        sess["db"].execute("DELETE FROM nodes WHERE created_at < ?", (cutoff,))
        sess["db"].commit()
        return {"pruned": True}


mcp = SessionManager(max_sessions=10)


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
        if self.path == "/status":
            self._json(200, {"status": "ok", "sessions": list(mcp.sessions.keys())})
        elif self.path == "/health":
            self._json(200, {"ok": True})
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
                r = mcp.search(chat_id, data.get("query", ""), data.get("top_k", 3))
                self._json(200, {"tool": tool, "results": r})
            elif tool == "ggm_insert":
                nid = mcp.insert(chat_id, data.get("concept", ""), data.get("content", ""), data.get("meta"))
                self._json(200, {"tool": tool, "node_id": nid})
            elif tool == "ggm_prune_session":
                r = mcp.prune(chat_id)
                self._json(200, {"tool": tool, **r})
            elif tool == "chat":
                query = data.get("user_message", "")
                ctx = mcp.search(chat_id, query, 2)
                self._json(200, {"tool": tool, "context": ctx})
            else:
                self._json(400, {"error": f"unknown tool: {tool}"})
        except Exception as e:
            self._json(500, {"error": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"MCP server on :{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
