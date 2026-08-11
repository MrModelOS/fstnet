#!/usr/bin/env python3
"""GGM Server — HTTP/MCP-демон графовой памяти для проектов.

Запуск:
  python3 ggm_server.py --port 8765 [--minilm]

API (HTTP JSON):
  POST /switch   {"path": "/proj"}     — переключить проект
  POST /search   {"query": "...", "top_k": 3}
  POST /reindex  {"path": "/proj"}     — перестроить граф
  GET  /status
  GET  /projects

Ollama: ollama create fstnet -f Modelfile
IDE:    context provider -> http://localhost:8765/search
"""
import os, sys, json, time, threading, pickle, argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class GGMServer:
    def __init__(self, use_minilm=True):
        self.use_minilm = use_minilm
        self.current_project = None
        self.current_ggm = None
        self.projects = {}
        self.lock = threading.Lock()

    def _make_ggm(self):
        from memory.ggm import GraphGatedMemory, GGMConfig, TfidfSvdEmbedder, MiniLMEmbedder
        cfg = GGMConfig()
        try:
            emb = MiniLMEmbedder() if self.use_minilm else TfidfSvdEmbedder(cfg.d_model)
        except Exception:
            emb = TfidfSvdEmbedder(cfg.d_model)
        return GraphGatedMemory(cfg, embedder=emb)

    def _idx_path(self, path):
        return os.path.join(path, ".ggm", "index.faiss")

    def _save(self, ggm, path):
        import faiss
        p = self._idx_path(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        faiss.write_index(ggm.index, p)
        meta = dict(node_registry=ggm.node_registry, file_chunks=ggm.file_chunks,
                    project_ids=ggm.project_ids, current_id=ggm.current_id,
                    graph_edges=list(ggm.graph.edges()))
        with open(p + ".meta", "wb") as f:
            pickle.dump(meta, f)
        return p

    def _load(self, path):
        import faiss
        p = self._idx_path(path)
        if not os.path.isfile(p) or not os.path.isfile(p + ".meta"):
            return None
        ggm = self._make_ggm()
        ggm.index = faiss.read_index(p)
        with open(p + ".meta", "rb") as f:
            meta = pickle.load(f)
        for k, v in meta.items():
            if k == "graph_edges":
                ggm.graph.add_nodes_from(range(ggm.current_id))
                ggm.graph.add_edges_from(v)
            else:
                setattr(ggm, k, v)
        return ggm

    def switch(self, path):
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return {"error": f"not found: {path}"}
        with self.lock:
            ggm = self._load(path)
            if ggm:
                self.current_ggm = ggm
                self.current_project = path
                self.projects[path] = ggm
                return {"status": "loaded", "project": path, "stats": ggm.stats()}
            ggm = self._make_ggm()
            ggm.index_directory([path], max_chunks_total=500_000)
            self._save(ggm, path)
            self.current_ggm = ggm
            self.current_project = path
            self.projects[path] = ggm
            return {"status": "indexed", "project": path, "stats": ggm.stats()}

    def reindex(self, path):
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return {"error": f"not found: {path}"}
        with self.lock:
            ggm = self._make_ggm()
            ggm.index_directory([path], max_chunks_total=500_000)
            self._save(ggm, path)
            self.projects[path] = ggm
            if self.current_project == path:
                self.current_ggm = ggm
            return {"status": "reindexed", "project": path, "stats": ggm.stats()}

    def search(self, query, top_k=3):
        with self.lock:
            if not self.current_ggm:
                return {"error": "no project loaded"}
            hits = self.current_ggm.query(query, top_k=top_k, graph_expand=2)
            ctx = self.current_ggm.recall_context(query, top_k=top_k)
            return {"project": self.current_project, "query": query,
                    "context": ctx, "hits": len(hits)}

    def status(self):
        return {"current": self.current_project,
                "projects": {p: g.stats() for p, g in self.projects.items()}}


# ── HTTP Handler ──────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_instance = None

    def log_message(self, *a):
        pass  # silence

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
        s = self.server_instance
        if self.path == "/status":
            self._json(200, s.status())
        elif self.path == "/projects":
            self._json(200, {"projects": list(s.projects.keys())})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        s = self.server_instance
        try:
            data = self._read()
        except Exception:
            self._json(400, {"error": "bad json"})
            return

        if self.path == "/switch":
            r = s.switch(data.get("path", ""))
            self._json(200, r)
        elif self.path == "/reindex":
            r = s.reindex(data.get("path", ""))
            self._json(200, r)
        elif self.path == "/search":
            r = s.search(data.get("query", ""), data.get("top_k", 3))
            self._json(200, r)
        else:
            self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--minilm", action="store_true", default=True)
    parser.add_argument("--no-minilm", action="store_true")
    args = parser.parse_args()

    use_minilm = args.minilm and not args.no_minilm
    ggm = GGMServer(use_minilm=use_minilm)
    Handler.server_instance = ggm

    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"GGM server on :{args.port} (minilm={use_minilm})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
