"""
core/vector_memory.py

Semantic memory / retrieval layer (the RAG substrate).

No network access to a model hub here, so no dense sentence-transformer
weights and no API embedding calls. Instead of mocking that out, this
uses a real local TF-IDF + cosine-similarity vector space (scikit-learn)
-- a legitimate classical sparse-embedding method.
The interface (`add`, `search`) is written so a dense encoder could
swap in later without touching callers -- see the `Embedder` protocol
below.

FAISS handles the similarity index itself over the TF-IDF projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


class Embedder(Protocol):
    def fit(self, corpus: List[str]) -> None: ...
    def transform(self, texts: List[str]) -> np.ndarray: ...


@dataclass
class MemoryRecord:
    doc_id: int
    text: str
    metadata: Dict[str, Any]


class TfidfEmbedder:
    """Real sparse-vector embedder: TF-IDF -> truncated SVD (LSA) to a
    fixed-dimensional dense space so it can be indexed by FAISS like any
    other embedding model."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
        self._svd: Optional[TruncatedSVD] = None
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        if not corpus:
            return
        tfidf = self._vectorizer.fit_transform(corpus)
        # TruncatedSVD needs at least 2 documents and n_components < n_features
        # to be well-defined; for tiny corpora fall back to raw (padded) TF-IDF.
        if tfidf.shape[0] < 2 or tfidf.shape[1] < 2:
            self._svd = None
            self._fitted = True
            return
        n_components = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        if self._svd is None:
            vecs = np.asarray(tfidf.todense()).astype("float32")
        else:
            vecs = self._svd.transform(tfidf).astype("float32")
        # pad/truncate to self.dim for a stable index dimension
        if vecs.shape[1] < self.dim:
            pad = np.zeros((vecs.shape[0], self.dim - vecs.shape[1]), dtype="float32")
            vecs = np.hstack([vecs, pad])
        elif vecs.shape[1] > self.dim:
            vecs = vecs[:, : self.dim]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class VectorMemory:
    """A real semantic retrieval store: embed -> index -> nearest-neighbor search."""

    def __init__(self, dim: int = 64, embedder: Optional[Embedder] = None):
        self.dim = dim
        self.embedder = embedder or TfidfEmbedder(dim=dim)
        self._records: List[MemoryRecord] = []
        self._matrix: Optional[np.ndarray] = None
        self._index = None

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        doc_id = len(self._records)
        self._records.append(MemoryRecord(doc_id=doc_id, text=text, metadata=metadata or {}))
        self._reindex()
        return doc_id

    def add_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        metadatas = metadatas or [{} for _ in texts]
        ids = []
        for t, m in zip(texts, metadatas):
            doc_id = len(self._records)
            self._records.append(MemoryRecord(doc_id=doc_id, text=t, metadata=m))
            ids.append(doc_id)
        self._reindex()
        return ids

    def _reindex(self) -> None:
        if not self._records:
            self._matrix = None
            self._index = None
            return
        texts = [r.text for r in self._records]
        self.embedder.fit(texts)
        vecs = self.embedder.transform(texts)
        self._matrix = vecs
        if _HAS_FAISS:
            index = faiss.IndexFlatIP(self.dim)
            index.add(vecs)
            self._index = index
        else:
            self._index = None  # fall back to numpy cosine search

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._records:
            return []
        qvec = self.embedder.transform([query])
        if self._index is not None:
            scores, idxs = self._index.search(qvec, min(k, len(self._records)))
            idxs, scores = idxs[0], scores[0]
        else:
            sims = (self._matrix @ qvec[0])
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]
        results = []
        for i, s in zip(idxs, scores):
            if i < 0:
                continue
            rec = self._records[int(i)]
            results.append({"doc_id": rec.doc_id, "text": rec.text,
                             "metadata": rec.metadata, "score": float(s)})
        return results

    def __len__(self) -> int:
        return len(self._records)
