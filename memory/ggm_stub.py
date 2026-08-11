import numpy as np
import faiss
import networkx as nx
from typing import List, Dict, Any, Optional


class GraphGatedMemory:
    """
    Graph-Gated Memory (GGM) для FST-Net.
    Векторный индекс в RAM (FAISS) + граф связей (NetworkX).
    Хранит факты, документы, контекст — не нагружая VRAM.
    """

    def __init__(self, d_model: int = 768, max_nodes: int = 500_000):
        self.d_model = d_model
        self.max_nodes = max_nodes

        self.index: faiss.Index = faiss.IndexFlatIP(d_model)
        self.graph = nx.DiGraph()
        self.node_registry: Dict[int, Dict[str, Any]] = {}
        self.current_id: int = 0

    def add_node(
        self,
        vector: np.ndarray,
        content: str,
        relations: Optional[List[int]] = None,
    ) -> int:
        if self.current_id >= self.max_nodes:
            raise RuntimeError(f"GGM limit reached: {self.max_nodes}")

        if vector.ndim > 1:
            vector = vector.flatten()[: self.d_model]

        norm_vec = vector.astype(np.float32)
        norm_vec /= np.linalg.norm(norm_vec) + 1e-9

        node_id = self.current_id
        self.index.add(np.expand_dims(norm_vec, axis=0))
        self.graph.add_node(node_id)

        self.node_registry[node_id] = {"content": content, "vector": norm_vec}

        if relations:
            for rel_id in relations:
                if self.graph.has_node(rel_id):
                    self.graph.add_edge(node_id, rel_id)

        self.current_id += 1
        return node_id

    def query(
        self, query_vector: np.ndarray, top_k: int = 4
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()[: self.d_model]

        q_vec = query_vector.astype(np.float32)
        q_vec /= np.linalg.norm(q_vec) + 1e-9

        scores, indices = self.index.search(np.expand_dims(q_vec, axis=0), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.node_registry:
                neighbors = list(self.graph.neighbors(idx))
                data = self.node_registry[idx].copy()
                data["neighbors"] = neighbors
                data["score"] = float(score)
                results.append(data)

        return results

    @property
    def size(self) -> int:
        return self.index.ntotal

    def index_texts(self, texts: List[str], embed_fn) -> int:
        """Индексирует список текстов через embed_fn(text) -> np.ndarray."""
        count = 0
        prev_id = None
        for text in texts:
            vec = embed_fn(text)
            if vec is not None:
                nid = self.add_node(vec, content=text, relations=[prev_id] if prev_id is not None else None)
                prev_id = nid
                count += 1
        return count
