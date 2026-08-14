"""Утилита для работы с GGM: индексация, сохранение/загрузка индекса, поиск.

Примеры:
  # Индексировать проект и сохранить
  python3 ggm_tool.py index /home/mrmodel/Проекты/MyAi/fstnet -o checkpoints/ggm_fstnet.faiss

  # Загрузить индекс и искать
  python3 ggm_tool.py search checkpoints/ggm_fstnet.faiss "how does training loop work"
"""
import os
import sys
import pickle
import faiss
import numpy as np


def save_ggm(ggm, path: str):
    """Сохраняет FAISS индекс + registry на диск."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    faiss.write_index(ggm.index, path)
    meta = {
        "node_registry": ggm.node_registry,
        "file_chunks": ggm.file_chunks,
        "project_ids": ggm.project_ids,
        "current_id": ggm.current_id,
        "graph_edges": list(ggm.graph.edges()),
        "stats": ggm.stats(),
    }
    with open(path + ".meta", "wb") as f:
        pickle.dump(meta, f)
    size = os.path.getsize(path) + os.path.getsize(path + ".meta")
    print(f"Saved GGM to {path} ({size/1024/1024:.1f} MB)")


def load_ggm(path: str, config=None, embedder=None):
    """Загружает GGM с диска."""
    from ggm import GraphGatedMemory, GGMConfig, TfidfSvdEmbedder
    if config is None:
        config = GGMConfig()
    ggm = GraphGatedMemory(config, embedder=embedder or TfidfSvdEmbedder(config.d_model))
    ggm.index = faiss.read_index(path)
    with open(path + ".meta", "rb") as f:
        meta = pickle.load(f)
    ggm.node_registry = meta["node_registry"]
    ggm.file_chunks = meta["file_chunks"]
    ggm.project_ids = meta["project_ids"]
    ggm.current_id = meta["current_id"]
    ggm.graph.add_nodes_from(range(ggm.current_id))
    ggm.graph.add_edges_from(meta["graph_edges"])
    st = ggm.stats()
    print(f"Loaded GGM: {st['nodes']:,} nodes, {st['projects']} projects")
    return ggm


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cmd = sys.argv[1]

    if cmd == "index":
        from ggm import GraphGatedMemory, GGMConfig, TfidfSvdEmbedder, MiniLMEmbedder
        target = sys.argv[2]
        out = "checkpoints/ggm_index.faiss"
        use_minilm = "--minilm" in sys.argv or "--minilm" in sys.argv
        args = [a for a in sys.argv[3:] if a not in ("--minilm", "-o") and not a.endswith(".faiss")]
        # parse -o
        if "-o" in sys.argv:
            idx = sys.argv.index("-o")
            if idx + 1 < len(sys.argv):
                out = sys.argv[idx + 1]
        cfg = GGMConfig()
        emb = MiniLMEmbedder() if use_minilm else TfidfSvdEmbedder(cfg.d_model)
        ggm = GraphGatedMemory(cfg, embedder=emb)
        roots = [target] if os.path.isdir(target) else [os.path.dirname(target)]
        ggm.index_directory(roots)
        save_ggm(ggm, out)

    elif cmd == "search":
        path = sys.argv[2]
        # default to MiniLM if available, fallback to TFIDF
        use_minilm = "--tfidf" not in sys.argv
        args = [a for a in sys.argv[3:] if a != "--tfidf"]
        query = " ".join(args)
        from ggm import TfidfSvdEmbedder, MiniLMEmbedder, GGMConfig
        cfg = GGMConfig()
        try:
            emb = MiniLMEmbedder() if use_minilm else TfidfSvdEmbedder(cfg.d_model)
        except Exception:
            emb = TfidfSvdEmbedder(cfg.d_model)
        ggm = load_ggm(path, embedder=emb)
        hits = ggm.query(query, top_k=4, graph_expand=2)
        for i, h in enumerate(hits):
            meta = h.get("meta", {})
            print(f"\n[{i+1}] score={h['score']:.3f} {meta.get('file','')} ({meta.get('project','')})")
            print(h["content"][:200].strip())

    elif cmd == "stats":
        path = sys.argv[2]
        with open(path + ".meta", "rb") as f:
            meta = pickle.load(f)
        st = meta["stats"]
        for k, v in st.items():
            print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
