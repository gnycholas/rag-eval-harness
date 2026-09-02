"""Both indexes, over the fixture corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.data import scifact
from rag_eval.index.base import rank
from rag_eval.index.sparse import Bm25Index, tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "scifact"


@pytest.fixture(scope="module")
def documents() -> list[scifact.Document]:
    return scifact.load(FIXTURE, check_counts=False).documents


@pytest.fixture(scope="module")
def bm25(documents: list[scifact.Document]) -> Bm25Index:
    index = Bm25Index()
    index.build(documents)
    return index


def test_tokenizer_keeps_alphanumeric_runs() -> None:
    assert tokenize("CD4+ T-cells, 2019!") == ["cd4", "t", "cells", "2019"]


def test_ties_break_on_document_id() -> None:
    ordered = rank({"b": 1.0, "a": 1.0, "c": 2.0}, k=3)
    assert [doc_id for doc_id, _ in ordered] == ["c", "a", "b"]


def test_zero_scores_are_dropped() -> None:
    assert rank({"a": 0.0, "b": 1.0}, k=5) == [("b", 1.0)]


def test_a_document_is_found_by_its_own_words(
    bm25: Bm25Index, documents: list[scifact.Document]
) -> None:
    target = documents[0]
    hits = bm25.search(target.title, k=5)
    assert target.doc_id in [doc_id for doc_id, _ in hits]


def test_a_query_with_no_known_term_returns_nothing(bm25: Bm25Index) -> None:
    assert bm25.search("zzzzqqqq wwwwvvvv", k=10) == []


def test_rare_terms_outweigh_common_ones(bm25: Bm25Index) -> None:
    """The reason to keep a sparse index next to a dense one."""
    common = bm25.search("the of and in a", k=5)
    assert not common or common[0][1] < 5.0


def test_search_is_deterministic(bm25: Bm25Index, documents: list[scifact.Document]) -> None:
    query = documents[3].title
    assert bm25.search(query, k=10) == bm25.search(query, k=10)


def test_rebuilding_gives_the_same_ranking(documents: list[scifact.Document]) -> None:
    first, second = Bm25Index(), Bm25Index()
    first.build(documents)
    second.build(documents)
    query = documents[5].title
    assert first.search(query, k=10) == second.search(query, k=10)


def test_k_limits_the_result(bm25: Bm25Index, documents: list[scifact.Document]) -> None:
    assert len(bm25.search(documents[0].indexed_text, k=3)) <= 3


def test_searching_before_building_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="has not been built"):
        Bm25Index().search("anything", k=5)


def test_dense_normalisation_makes_dot_product_cosine() -> None:
    """Checked without loading a model, so the suite stays offline."""
    import numpy as np

    from rag_eval.index.dense import _normalize

    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    unit = _normalize(vectors)

    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
    assert unit[0] @ unit[0] == pytest.approx(1.0)


def test_dense_normalisation_survives_a_zero_vector() -> None:
    import numpy as np

    from rag_eval.index.dense import _normalize

    assert not np.isnan(_normalize(np.zeros((1, 3), dtype=np.float32))).any()


def test_a_dense_index_round_trips_through_disk(tmp_path: Path) -> None:
    import numpy as np

    from rag_eval.index.dense import DenseIndex

    index = DenseIndex(model_name="pretend/model")
    index.doc_ids = ["a", "b"]
    index._vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    path = tmp_path / "dense.npz"
    index.save(path)
    restored = DenseIndex.load(path)

    assert restored.doc_ids == ["a", "b"]
    assert restored.model_name == "pretend/model"
    assert restored.dimension == 2


def test_searching_an_unbuilt_dense_index_is_an_error() -> None:
    from rag_eval.index.dense import DenseIndex

    with pytest.raises(RuntimeError, match="has not been built"):
        DenseIndex().search("anything", k=5)


def test_an_index_built_by_another_model_is_refused(tmp_path: Path) -> None:
    """Loading one model's vectors and querying them with another's returns
    confident nonsense and raises nothing."""
    import numpy as np

    from rag_eval.index.dense import DenseIndex

    index = DenseIndex(model_name="model/a")
    index.doc_ids = ["x"]
    index._vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    path = tmp_path / "dense.npz"
    index.save(path)

    with pytest.raises(ValueError, match="built with 'model/a'"):
        DenseIndex.load(path, expect_model="model/b")


def test_loading_without_an_expectation_still_works(tmp_path: Path) -> None:
    import numpy as np

    from rag_eval.index.dense import DenseIndex

    index = DenseIndex(model_name="model/a")
    index.doc_ids = ["x"]
    index._vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    path = tmp_path / "dense.npz"
    index.save(path)

    assert DenseIndex.load(path).model_name == "model/a"
