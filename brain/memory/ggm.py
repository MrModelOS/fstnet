"""GGM (Graph-Gated Memory) для FST-Net.

Полноценная графовая память: FAISS (вектора) + NetworkX (связи).

Эмбеддер (по умолчанию): TF-IDF + TruncatedSVD — полностью офлайн,
быстрый, на основе собственного корпуса. Опционально — MiniLM-L6
(семантические эмбеддинги) при наличии скачанной модели.

Индексация локальных проектов/документации в RAM:
  - Чтение файлов исходного кода и документации
  - Разбивка на чанки с перекрытием
  - Графовые связи: соседние чанки одного файла, файлы проекта
  - Векторный индекс FAISS + поиск с графовым расширением
"""
import os
import numpy as np
import faiss
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GGMConfig:
    d_model: int = 384
    max_nodes: int = 1_000_000
    chunk_chars: int = 600
    overlap: int = 80
    max_neighbors: int = 4


# расширения для индексации
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".zsh",
    ".sql", ".css", ".scss", ".html", ".json", ".yaml", ".yml",
    ".toml", ".xml", ".md", ".txt", ".rst", ".ini", ".cfg",
    ".lua", ".kt", ".swift", ".sol", ".vy",
}
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist",
    "build", ".next", ".cache", "target", "checkpoints", "data",
    ".idea", ".vscode", "logs", "coverage", ".tox", ".mypy_cache",
    "vendor", "out", "bin", ".gradle", ".cargo", "snap", ".local",
    "snapshots", "blobs", "refs", "hub",
}
SKIP_FILES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "Cargo.lock", "go.sum", "CACHEDIR.TAG",
}
# файлы без расширений, которые индексируем как текст
NOEXT_TEXT_FILES = {
    "README", "LICENSE", "COPYING", "CHANGELOG", "Makefile", "Dockerfile",
    "rust-book", "git-docs", "node-docs", "react-docs", "docker-docs",
    "linux-kernel", "rust", "python", "go",
}
MAX_FILE_BYTES = 1_500_000


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class TfidfSvdEmbedder:
    """TF-IDF + TruncatedSVD — офлайн, быстрый, строится на корпусе проектов."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._tfidf = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self._tfidf = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w[\w.]+\b|[^\s\d\w]{2,}",
            sublinear_tf=True,
        )
        n_components = min(self.dim, max(64, len(texts) // 3))
        self._svd = make_pipeline(
            TruncatedSVD(n_components=n_components, random_state=42),
            Normalizer(copy=False),
        )
        tfidf_mat = self._tfidf.fit_transform(texts)
        self._svd.fit(tfidf_mat)
        self._fitted = True
        return self

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        from scipy.sparse import issparse
        mat = self._tfidf.transform([text])
        vec = self._svd.transform(mat)[0].astype(np.float32)
        if vec.shape[0] < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.shape[0]))
        elif vec.shape[0] > self.dim:
            vec = vec[: self.dim]
        return vec

    def embed_many(self, texts: List[str]) -> np.ndarray:
        if not self._fitted or not texts:
            return np.zeros((len(texts), self.dim), dtype=np.float32)
        mats = self._tfidf.transform(texts)
        vecs = self._svd.transform(mats).astype(np.float32)
        if vecs.shape[1] < self.dim:
            vecs = np.pad(vecs, ((0, 0), (0, self.dim - vecs.shape[1])))
        elif vecs.shape[1] > self.dim:
            vecs = vecs[:, : self.dim]
        return vecs

    def upgrade_to_minilm(self) -> bool:
        """Пытается переключиться на MiniLM если модель доступна."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            cache = os.path.expanduser(
                "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
            )
            model_dir = None
            if os.path.isdir(cache):
                for d in os.listdir(cache):
                    cand = os.path.join(cache, d)
                    if os.path.isfile(os.path.join(cand, "model.safetensors")):
                        model_dir = cand
                        break
            if not model_dir:
                return False
            tok = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModel.from_pretrained(model_dir)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            self._minilm = model
            self._minilm_tok = tok
            self._minilm_device = device
            self._use_minilm = True
            print(f"GGM: upgraded to MiniLM on {device}", flush=True)
            return True
        except Exception:
            return False


class MiniLMEmbedder:
    """MiniLM-L6-v2 через transformers (384-dim, семантический)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.dim = 384
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.torch = torch
        path = "sentence-transformers/all-MiniLM-L6-v2"
        self._tok = AutoTokenizer.from_pretrained(path)
        self._model = AutoModel.from_pretrained(path)
        self._model.to(self.device)
        self._model.eval()

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        import torch
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                enc = self._tok(
                    batch, padding=True, truncation=True, max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self._model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                vecs.append(pooled.cpu().numpy().astype(np.float32))
        allv = np.concatenate(vecs, axis=0)
        norms = np.linalg.norm(allv, axis=1, keepdims=True)
        return allv / np.clip(norms, 1e-9, None)


# ---------------------------------------------------------------------------
# GGM
# ---------------------------------------------------------------------------

class GraphGatedMemory:
    """Графовая память: FAISS + NetworkX."""

    def __init__(self, config: Optional[GGMConfig] = None, embedder=None):
        self.config = config or GGMConfig()
        self.embedder = embedder or TfidfSvdEmbedder(self.config.d_model)

        self.index = faiss.IndexFlatIP(self.config.d_model)
        self.graph = nx.DiGraph()
        self.node_registry: Dict[int, Dict[str, Any]] = {}
        self.current_id: int = 0
        self.file_chunks: Dict[Tuple[str, str], List[int]] = {}
        self.project_ids: Dict[str, List[int]] = {}
        self._fitted = isinstance(self.embedder, TfidfSvdEmbedder) and self.embedder._fitted

    # ----- add -----

    def add_node(
        self,
        vector: np.ndarray,
        content: str,
        relations: Optional[List[int]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        if self.current_id >= self.config.max_nodes:
            return -1

        vec = vector.flatten().astype(np.float32)
        if vec.shape[0] != self.config.d_model:
            if vec.shape[0] > self.config.d_model:
                vec = vec[: self.config.d_model]
            else:
                vec = np.pad(vec, (0, self.config.d_model - vec.shape[0]))
        n = np.linalg.norm(vec)
        vec = vec / n if n > 0 else vec

        node_id = self.current_id
        self.index.add(np.expand_dims(vec, axis=0))
        self.graph.add_node(node_id)
        self.node_registry[node_id] = {
            "content": content, "vector": vec, "meta": meta or {},
        }
        added = 0
        if relations:
            for rel_id in relations:
                if self.graph.has_node(rel_id) and rel_id != node_id:
                    self.graph.add_edge(node_id, rel_id)
                    self.graph.add_edge(rel_id, node_id)
                    added += 1
                    if added >= self.config.max_neighbors:
                        break
        self.current_id += 1
        return node_id

    # ----- indexing -----

    @staticmethod
    def _walk_files(roots: List[str]) -> List[Tuple[str, str]]:
        out = []
        for root in roots:
            root = os.path.abspath(root)
            if not os.path.isdir(root):
                if os.path.isfile(root):
                    out.append((os.path.basename(os.path.dirname(root) or root), root))
                continue
            project = os.path.basename(root) or root
            for dp, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
                for f in sorted(files):
                    if f in SKIP_FILES or f.startswith("."):
                        continue
                    ext = os.path.splitext(f)[1].lower()
                    base = os.path.splitext(f)[0]
                    if ext not in CODE_EXTS and base not in NOEXT_TEXT_FILES:
                        continue
                    p = os.path.join(dp, f)
                    try:
                        if os.path.getsize(p) > MAX_FILE_BYTES:
                            continue
                    except OSError:
                        continue
                    out.append((project, p))
        return out

    @staticmethod
    def _read_text(path: str) -> str:
        for enc in ("utf-8", "latin-1", "cp1251"):
            try:
                with open(path, "r", encoding=enc, errors="replace") as fh:
                    return fh.read()
            except (UnicodeDecodeError, OSError):
                continue
        return ""

    def _chunks(self, text: str) -> List[str]:
        c, ov = self.config.chunk_chars, self.config.overlap
        lines = text.splitlines(keepends=True)
        res, cur = [], ""
        for line in lines:
            if len(cur) + len(line) > c and cur:
                res.append(cur)
                tail = cur[-ov:] if ov > 0 else ""
                cur = tail + line
            else:
                cur += line
        if cur.strip():
            res.append(cur)
        return res or [text]

    def index_directory(
        self,
        roots: List[str],
        max_chunks_total: int = 300_000,
        progress: bool = True,
    ) -> int:
        """Индексирует проекты/документацию. Возвращает число узлов."""
        files = self._walk_files(roots)
        if progress:
            print(f"GGM: found {len(files)} indexable files", flush=True)

        # собираем все чанки
        all_contents: List[str] = []
        all_meta: List[Dict[str, Any]] = []
        file_boundaries: List[Tuple[int, int]] = []  # (start_idx, end_idx) для каждого файла
        total = 0

        for project, path in files:
            text = self._read_text(path)
            if not text.strip():
                continue
            chunk_texts = self._chunks(text)
            start = total
            for ci, chunk in enumerate(chunk_texts):
                if total >= max_chunks_total:
                    break
                all_contents.append(chunk)
                all_meta.append({
                    "project": project, "path": path, "chunk": ci,
                    "file": os.path.basename(path),
                })
                total += 1
            if total > start:
                file_boundaries.append((project, path, start, total))

        if progress:
            print(f"GGM: collected {total:,} chunks, embedding...", flush=True)

        # обучаем TF-IDF на всём корпусе если нужно
        if isinstance(self.embedder, TfidfSvdEmbedder) and not self.embedder._fitted:
            self.embedder.fit(all_contents)

        # считаем эмбеддинги батчами
        if isinstance(self.embedder, TfidfSvdEmbedder):
            vecs = self.embedder.embed_many(all_contents)
        else:
            vecs = self.embedder.embed_many(all_contents)

        # добавляем узлы с графами связей
        # строим карту позиция -> (start, end) файла для O(1) поиска
        pos_to_file: Dict[int, Tuple[int, int]] = {}
        for proj, path, s, e in file_boundaries:
            for idx in range(s, e):
                pos_to_file[idx] = (s, e)

        for i, (content, meta) in enumerate(zip(all_contents, all_meta)):
            s, e = pos_to_file.get(i, (i, i + 1))
            relations = []
            if i - 1 >= s:
                relations.append(i - 1)
            if i + 1 < e:
                relations.append(i + 1)
            self.add_node(vecs[i], content=content, relations=relations, meta=meta)

        # запоминаем границы файлов
        for proj, path, s, e in file_boundaries:
            ids = list(range(s, e))
            self.file_chunks[(proj, path)] = ids
            self.project_ids.setdefault(proj, []).extend(ids)

        if progress:
            print(f"GGM: indexed {total:,} chunks from {len(files)} files", flush=True)
        return total

    # ----- query -----

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        graph_expand: int = 2,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        q_vec = self.embedder.embed(query_text)
        if q_vec.ndim == 1:
            q_vec = np.expand_dims(q_vec, axis=0)
        q_vec = q_vec.astype(np.float32)
        norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_vec = q_vec / np.clip(norms, 1e-9, None)

        scores, indices = self.index.search(
            q_vec, min(top_k * 4, self.index.ntotal)
        )
        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx in seen:
                continue
            seen.add(idx)
            data = self.node_registry.get(idx)
            if not data:
                continue
            results.append({"node_id": idx, "score": float(score), **data})

        expanded = list(results)
        if graph_expand > 0:
            for r in results[:top_k]:
                for nb in self.graph.neighbors(r["node_id"]):
                    if nb in seen:
                        continue
                    seen.add(nb)
                    nd = self.node_registry.get(nb)
                    if nd:
                        expanded.append({"node_id": nb, "score": r["score"] * 0.8, **nd})

        expanded.sort(key=lambda x: x["score"], reverse=True)
        dedup, seen_sig = [], set()
        for r in expanded:
            sig = r["content"][:80]
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            dedup.append(r)
            if len(dedup) >= top_k * 2:
                break
        return dedup[: top_k * 2]

    def recall_context(self, query_text: str, top_k: int = 3) -> str:
        hits = self.query(query_text, top_k=top_k, graph_expand=1)
        if not hits:
            return ""
        parts = []
        for h in hits:
            meta = h.get("meta", {})
            loc = meta.get("path", "")
            if loc:
                parts.append(f"### {os.path.basename(loc)} — {meta.get('project','')}")
            parts.append(h["content"].strip())
        return "\n\n".join(parts)[:3500]

    # ----- status -----

    @property
    def size(self) -> int:
        return self.index.ntotal

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.index.ntotal,
            "projects": len(self.project_ids),
            "files": len(self.file_chunks),
            "edges": self.graph.number_of_edges(),
            "embedder": type(self.embedder).__name__,
        }
