#!/usr/bin/env python3
"""GGM MCP Server — Model Context Protocol для графовой памяти.

Подключение в OpenCode / Continue.dev / Cursor:
  .opencodemcp.json:
  {
    "mcpServers": {
      "ggm": {
        "command": "python3",
        "args": ["/path/to/fstnet/ggm_mcp.py"]
      }
    }
  }

Инструменты:
  switch_project(path) — переключить граф на проект
  search_context(query, top_k=3) — поиск контекста в графе
  reindex_project(path) — перестроить индекс проекта
"""
import sys, json, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggm_server import GGMServer

ggm = GGMServer(use_minilm=True)


def handle(req):
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {"name": "switch_project",
                 "description": "Switch graph to project",
                 "inputSchema": {"type": "object",
                                 "properties": {"path": {"type": "string"}},
                                 "required": ["path"]}},
                {"name": "search_context",
                 "description": "Search context in project graph",
                 "inputSchema": {"type": "object",
                                 "properties": {"query": {"type": "string"},
                                                "top_k": {"type": "number"}},
                                 "required": ["query"]}},
                {"name": "reindex_project",
                 "description": "Rebuild project index",
                 "inputSchema": {"type": "object",
                                 "properties": {"path": {"type": "string"}},
                                 "required": ["path"]}},
            ]
        }

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if name == "switch_project":
                r = ggm.switch(args["path"])
            elif name == "search_context":
                r = ggm.search(args.get("query", ""), args.get("top_k", 3))
            elif name == "reindex_project":
                r = ggm.reindex(args["path"])
            else:
                return {"error": {"code": -1, "message": f"unknown: {name}"}}
            return {"content": [{"type": "text", "text": json.dumps(r, ensure_ascii=False)}]}
        except Exception as e:
            return {"error": {"code": -1, "message": str(e)}}

    return {"error": {"code": -32601, "message": f"unknown method: {method}"}}


def main():
    # MCP over stdio (JSON-RPC 2.0)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            req = json.loads(line)
            resp = handle(req)
            resp["jsonrpc"] = "2.0"
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            sys.stderr.write(f"MCP error: {e}\n")


if __name__ == "__main__":
    main()
