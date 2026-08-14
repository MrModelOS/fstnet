#!/usr/bin/env python3
"""STDM - State-Tree Delta Memory.

Builds AST tree of project, stores in SQLite, generates delta prompts.
MCP injects only changed nodes (~150 tokens) instead of full files.
"""
import ast, os, json, sqlite3, hashlib, time
from pathlib import Path

STDM_DB = "data/ggm/stdm.db"


class STDM:
    def __init__(self):
        self.db = sqlite3.connect(STDM_DB, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY, file TEXT, name TEXT, kind TEXT,
            signature TEXT, body_hash TEXT, body TEXT, parent TEXT,
            created_at REAL, updated_at REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS edges (
            src TEXT, dst TEXT, kind TEXT, PRIMARY KEY(src,dst))""")
        self.db.commit()

    def index_file(self, filepath):
        """Parse Python file and store AST nodes."""
        try:
            source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            return 0

        file_id = hashlib.md5(filepath.encode()).hexdigest()[:12]
        count = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._func_signature(node, source)
                body = ast.get_source_segment(source, node) or ""
                nid = f"{file_id}_func_{node.name}"
                body_hash = hashlib.md5(body.encode()).hexdigest()

                existing = self.db.execute("SELECT body_hash FROM nodes WHERE id=?", (nid,)).fetchone()
                if existing and existing[0] == body_hash:
                    continue  # unchanged

                self.db.execute(
                    "INSERT OR REPLACE INTO nodes (id,file,name,kind,signature,body_hash,body,parent,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (nid, filepath, node.name, "function", sig, body_hash, body, file_id, time.time(), time.time())
                )
                count += 1

                # Index calls inside this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        edge_dst = f"func_{child.func.id}"
                        self.db.execute("INSERT OR IGNORE INTO edges (src,dst,kind) VALUES (?,?,?)", (nid, edge_dst, "calls"))

            elif isinstance(node, ast.ClassDef):
                sig = f"class {node.name}"
                bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, 'unparse') else []
                if bases:
                    sig += f"({', '.join(bases)})"
                nid = f"{file_id}_class_{node.name}"
                body = ast.get_source_segment(source, node) or ""
                body_hash = hashlib.md5(body.encode()).hexdigest()

                self.db.execute(
                    "INSERT OR REPLACE INTO nodes (id,file,name,kind,signature,body_hash,body,parent,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (nid, filepath, node.name, "class", sig, body_hash, body, file_id, time.time(), time.time())
                )
                count += 1

        self.db.commit()
        return count

    def _func_signature(self, node, source):
        args = []
        for arg in node.args.args:
            name = arg.arg
            annotation = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') and arg.annotation else ""
            args.append(f"{name}: {annotation}" if annotation else name)
        ret = f" -> {ast.unparse(node.returns)}" if hasattr(ast, 'unparse') and node.returns else ""
        return f"def {node.name}({', '.join(args)}){ret}"

    def index_directory(self, root, ext=".py"):
        total = 0
        for dp, _, files in os.walk(root):
            for f in files:
                if f.endswith(ext):
                    total += self.index_file(os.path.join(dp, f))
        return total

    def get_delta(self, filepath, old_hash=None):
        """Get only changed nodes since last check."""
        rows = self.db.execute(
            "SELECT name, signature, body, updated_at FROM nodes WHERE file=? AND updated_at > COALESCE(?, 0)",
            (filepath, old_hash or 0)
        ).fetchall()
        return [{"name": r[0], "signature": r[1], "body": r[2], "updated": r[3]} for r in rows]

    def get_context_for(self, query, top_k=3):
        """Find relevant AST nodes for a query."""
        results = []
        # Search by name match
        rows = self.db.execute(
            "SELECT name, signature, body FROM nodes WHERE name LIKE ? OR signature LIKE ? LIMIT ?",
            (f"%{query}%", f"%{query}%", top_k)
        ).fetchall()
        for r in rows:
            results.append({"name": r[0], "signature": r[1], "body": r[2][:300]})
        return results

    def get_callers(self, func_name):
        """Find who calls this function."""
        rows = self.db.execute(
            "SELECT n.name, n.signature FROM edges e JOIN nodes n ON e.src = n.id WHERE e.dst = ?",
            (f"func_{func_name}",)
        ).fetchall()
        return [{"name": r[0], "signature": r[1]} for r in rows]

    def stats(self):
        nodes = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        files = self.db.execute("SELECT COUNT(DISTINCT file) FROM nodes").fetchone()[0]
        return {"nodes": nodes, "edges": edges, "files": files}


if __name__ == "__main__":
    s = STDM()
    count = s.index_directory("/home/mrmodel/Проекты/MyAi/fstnet")
    print(f"Indexed {count} nodes from fstnet")
    print(s.stats())
