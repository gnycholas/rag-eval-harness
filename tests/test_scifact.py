"""Reading the dataset, over a fixture cut from the real files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.data import scifact
from rag_eval.data.scifact import CONTRADICT, SUPPORT, DatasetError

FIXTURE = Path(__file__).parent / "fixtures" / "scifact"


@pytest.fixture(scope="module")
def dataset() -> scifact.Dataset:
    return scifact.load(FIXTURE, check_counts=False)


def test_documents_carry_title_and_text(dataset: scifact.Dataset) -> None:
    document = dataset.documents[0]
    assert document.doc_id
    assert document.indexed_text.startswith(document.title)


def test_indexed_text_survives_a_missing_title() -> None:
    assert scifact.Document("1", "", "body").indexed_text == "body"


def test_human_verdicts_are_read_off_the_query(dataset: scifact.Dataset) -> None:
    verdicts = {q.verdict for q in dataset.queries if q.verdict}
    assert verdicts <= {SUPPORT, CONTRADICT}
    assert verdicts


def test_a_query_without_a_label_is_kept_with_none(dataset: scifact.Dataset) -> None:
    assert any(q.verdict is None for q in dataset.queries)


def test_conflicting_human_labels_are_refused() -> None:
    """Averaging a disagreement away would hide a labelling problem."""
    metadata = {
        "1": [{"label": SUPPORT}],
        "2": [{"label": CONTRADICT}],
    }
    with pytest.raises(DatasetError, match="conflicting human labels"):
        scifact._verdict_of(metadata)


def test_qrels_map_query_to_relevant_documents(dataset: scifact.Dataset) -> None:
    _, relevant = next(iter(dataset.qrels.items()))
    assert relevant
    assert all(isinstance(score, int) for score in relevant.values())


def test_only_judged_queries_count(dataset: scifact.Dataset) -> None:
    judged = dataset.judged()
    assert judged
    assert all(q.query_id in dataset.qrels for q in judged)


def test_queries_with_a_verdict_are_a_subset_of_judged(dataset: scifact.Dataset) -> None:
    assert set(dataset.with_verdict()) <= set(dataset.judged())


def test_wrong_counts_are_refused() -> None:
    """If BEIR republishes different contents, every README number silently
    starts describing something else."""
    with pytest.raises(DatasetError, match="expected 5183 documents"):
        scifact.load(FIXTURE, check_counts=True)


def test_a_response_that_is_not_a_zip_is_refused() -> None:
    with pytest.raises(DatasetError, match="not a zip archive"):
        scifact._reject_if_not_a_zip(b"<!DOCTYPE html><html>nope</html>")


def test_a_real_zip_passes() -> None:
    scifact._reject_if_not_a_zip(b"PK\x03\x04rest")


def test_download_skips_when_already_present(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "scifact"
    target.mkdir()
    (target / "corpus.jsonl").write_text("", encoding="utf-8")
    (target / ".sha256").write_text("deadbeef", encoding="utf-8")

    def explode(url: str) -> bytes:
        raise AssertionError("a skip must not touch the network")

    monkeypatch.setattr(scifact, "_fetch", explode)
    assert scifact.download(target) == target


def test_a_zip_with_a_traversing_path_is_refused(tmp_path: Path, monkeypatch) -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.txt", "no")

    monkeypatch.setattr(scifact, "_fetch", lambda url: buffer.getvalue())
    with pytest.raises(DatasetError, match="refusing to extract"):
        scifact.download(tmp_path / "scifact", force=True)


def test_the_fixture_is_real_data() -> None:
    """Invented fixtures do not reproduce sparse relevance or rare terms."""
    record = json.loads((FIXTURE / "corpus.jsonl").read_text().splitlines()[0])
    assert record["_id"].isdigit()
    assert len(record["text"]) > 200
