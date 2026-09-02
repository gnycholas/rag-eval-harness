"""What every index has to provide."""

from __future__ import annotations

from typing import Protocol

Hit = tuple[str, float]


class Index(Protocol):
    """Search over the corpus.

    Ties are broken by document id, not left to sort order. Without that the
    ranking wobbles between runs and the determinism test fails for the wrong
    reason.
    """

    def search(self, query: str, k: int) -> list[Hit]: ...


def rank(scores: dict[str, float], k: int) -> list[Hit]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, score) for doc_id, score in ordered[:k] if score > 0]
