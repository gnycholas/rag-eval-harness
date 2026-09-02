"""Embedding search over the corpus.

Embeddings run locally through ONNX, so the index needs no API key and no GPU.
Which model is a measured choice, not a reputation one: see the comparison in
the README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from rag_eval.data.scifact import Document
from rag_eval.index.base import Hit, rank

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 256

log = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit length, so a dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized: np.ndarray = vectors / np.maximum(norms, 1e-12)
    return normalized


@dataclass
class DenseIndex:
    model_name: str = DEFAULT_MODEL
    doc_ids: list[str] = field(default_factory=list)
    _vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.size else 0

    def build(self, documents: list[Document]) -> None:
        self.doc_ids = [d.doc_id for d in documents]
        texts = [d.indexed_text for d in documents]
        embedded = np.asarray(
            list(self.model.embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32
        )
        self._vectors = _normalize(embedded)
        log.info("embedded %d documents into %d dimensions", len(documents), self.dimension)

    def search(self, query: str, k: int) -> list[Hit]:
        if not self._vectors.size:
            raise RuntimeError("index has not been built")

        embedded = np.asarray(list(self.model.embed([query])), dtype=np.float32)
        similarity = self._vectors @ _normalize(embedded)[0]
        # Cosine runs to -1; shift so that rank()'s "drop non-positive" rule
        # does not silently discard the whole tail.
        shifted = (similarity + 1.0) / 2.0
        return rank(dict(zip(self.doc_ids, shifted.tolist(), strict=True)), k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, vectors=self._vectors, doc_ids=np.array(self.doc_ids), model=self.model_name
        )

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        stored = np.load(path, allow_pickle=False)
        index = cls(model_name=str(stored["model"]))
        index.doc_ids = [str(x) for x in stored["doc_ids"]]
        index._vectors = stored["vectors"]
        return index
