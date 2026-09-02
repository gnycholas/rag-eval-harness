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

# Measured on the 809 training queries rather than chosen by reputation.
# Quality, with intervals that do not overlap:
#   bge-small-en-v1.5   nDCG@10 0.7522 [0.7270, 0.7754]
#   all-MiniLM-L6-v2    nDCG@10 0.6387 [0.6096, 0.6666]
#   arctic-embed-s      nDCG@10 0.6002 [0.5711, 0.6278]
# Throughput, measured head to head on the same 200 documents:
#   all-MiniLM-L6-v2     11.8 ms/doc
#   bge-small-en-v1.5   670.8 ms/doc
# The default is the fast one: 0.11 nDCG costs a 57x longer first build,
# about an hour against a minute, and an hour before anything appears is
# how a repository ends up never being run. HIGH_QUALITY_MODEL is one flag
# away and the README carries both numbers.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HIGH_QUALITY_MODEL = "BAAI/bge-small-en-v1.5"
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
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[Hit]]:
        """One embedding call for the whole batch.

        Per-call overhead dominates otherwise: scoring a split one query at a
        time took tens of minutes and made the harness impractical to run.
        """
        if not self._vectors.size:
            raise RuntimeError("index has not been built")

        embedded = np.asarray(
            list(self.model.embed(queries, batch_size=BATCH_SIZE)), dtype=np.float32
        )
        similarity = self._vectors @ _normalize(embedded).T

        # Cosine runs to -1; shift so that rank()'s "drop non-positive" rule
        # does not silently discard the whole tail.
        shifted = (similarity + 1.0) / 2.0
        return [
            rank(dict(zip(self.doc_ids, shifted[:, column].tolist(), strict=True)), k)
            for column in range(shifted.shape[1])
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, vectors=self._vectors, doc_ids=np.array(self.doc_ids), model=self.model_name
        )

    @classmethod
    def load(cls, path: Path, *, expect_model: str | None = None) -> DenseIndex:
        stored = np.load(path, allow_pickle=False)
        model_name = str(stored["model"])

        # Loading vectors built by one model and querying them with another
        # returns confident nonsense and raises nothing.
        if expect_model and model_name != expect_model:
            raise ValueError(f"index was built with {model_name!r}, not {expect_model!r}")

        index = cls(model_name=model_name)
        index.doc_ids = [str(x) for x in stored["doc_ids"]]
        index._vectors = stored["vectors"]
        return index
