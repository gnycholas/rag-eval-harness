"""BM25 over title and text.

Written here rather than pulled in: rank-bm25 has sat at 0.2.2 for years, the
scoring is worth keeping auditable, and the tie-breaking has to be ours for the
determinism test to mean anything.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from rag_eval.data.scifact import Document
from rag_eval.index.base import Hit, rank

# Okapi defaults. Tuning them is an ablation, not a constant to pick by feel.
K1 = 1.2
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs.

    No stemming. It costs a little recall against a Lucene style baseline, and
    the README says so rather than quietly comparing an apple to an orange.
    """
    return _TOKEN.findall(text.lower())


@dataclass
class Bm25Index:
    k1: float = K1
    b: float = B
    doc_ids: list[str] = field(default_factory=list)
    vocabulary: dict[str, int] = field(default_factory=dict)
    _matrix: csr_matrix | None = None
    _idf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _length_norm: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def build(self, documents: list[Document]) -> None:
        self.doc_ids = [d.doc_id for d in documents]
        tokenized = [tokenize(d.indexed_text) for d in documents]

        self.vocabulary = {}
        rows, cols, values = [], [], []
        for row, tokens in enumerate(tokenized):
            for term, count in Counter(tokens).items():
                col = self.vocabulary.setdefault(term, len(self.vocabulary))
                rows.append(row)
                cols.append(col)
                values.append(count)

        shape = (len(documents), len(self.vocabulary))
        self._matrix = csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float32)

        lengths = np.asarray([len(t) for t in tokenized], dtype=np.float32)
        average = float(lengths.mean()) if len(lengths) else 0.0
        self._length_norm = self.k1 * (1 - self.b + self.b * lengths / (average or 1.0))

        document_frequency = np.asarray((self._matrix > 0).sum(axis=0)).ravel()
        total = len(documents)
        self._idf = np.log(
            1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)

    def search(self, query: str, k: int) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("index has not been built")

        columns = [self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary]
        if not columns:
            return []

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        for column in columns:
            frequencies = self._matrix[:, column].toarray().ravel()
            present = frequencies > 0
            if not present.any():
                continue
            numerator = frequencies[present] * (self.k1 + 1)
            denominator = frequencies[present] + self._length_norm[present]
            scores[present] += self._idf[column] * numerator / denominator

        return rank(dict(zip(self.doc_ids, scores.tolist(), strict=True)), k)
