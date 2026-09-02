"""Fetching and reading the SciFact split of BEIR.

The counts are checked after extraction. If BEIR ever republishes the dataset
with different contents, every number in the README silently starts describing
something else; better to fail.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
TIMEOUT = 180
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0

EXPECTED_DOCS = 5183
EXPECTED_QUERIES = 1109

SUPPORT = "SUPPORT"
CONTRADICT = "CONTRADICT"
NOINFO = "NOINFO"
VERDICTS = (SUPPORT, CONTRADICT, NOINFO)

log = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """The dataset is not what we expect."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def indexed_text(self) -> str:
        return f"{self.title} {self.text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    verdict: str | None


@dataclass(frozen=True)
class Dataset:
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]

    @property
    def by_id(self) -> dict[str, Document]:
        return {d.doc_id: d for d in self.documents}

    def judged(self) -> list[Query]:
        """Queries this split has human relevance judgments for."""
        return [q for q in self.queries if q.query_id in self.qrels]

    def with_verdict(self) -> list[Query]:
        return [q for q in self.judged() if q.verdict is not None]


def _fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)
    raise DatasetError(f"giving up after {MAX_ATTEMPTS} attempts: {last}")


def _reject_if_not_a_zip(payload: bytes) -> None:
    """A server that answers 200 with an HTML error page is a real failure mode."""
    if payload[:2] != b"PK":
        head = payload[:80]
        raise DatasetError(f"response is not a zip archive: {head!r}")


def download(target: Path, *, force: bool = False) -> Path:
    """Download and extract into target, skipping when the digest still matches."""
    target.mkdir(parents=True, exist_ok=True)
    digest_file = target / ".sha256"

    if not force and (target / "corpus.jsonl").exists() and digest_file.exists():
        log.info("dataset already present, skipping download")
        return target

    payload = _fetch(URL)
    _reject_if_not_a_zip(payload)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            name = Path(member)
            if name.is_absolute() or ".." in name.parts:
                raise DatasetError(f"refusing to extract {member!r}")
        archive.extractall(target.parent)

    digest_file.write_text(hashlib.sha256(payload).hexdigest(), encoding="utf-8")
    log.info("dataset extracted to %s", target)
    return target


def _read_documents(path: Path) -> list[Document]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            documents.append(
                Document(record["_id"], record.get("title", ""), record.get("text", ""))
            )
    return documents


def _verdict_of(metadata: dict[str, object]) -> str | None:
    """The human label carried on the query, when there is one.

    A claim can be judged against more than one document. They agree in this
    dataset, and a disagreement would be a labelling problem rather than
    something to average away, so it is refused.
    """
    labels = {
        entry["label"]
        for evidences in metadata.values()
        if isinstance(evidences, list)
        for entry in evidences
        if isinstance(entry, dict) and entry.get("label")
    }
    if not labels:
        return None
    if len(labels) > 1:
        raise DatasetError(f"conflicting human labels on one claim: {sorted(labels)}")
    return str(labels.pop())


def _read_queries(path: Path) -> list[Query]:
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            queries.append(
                Query(record["_id"], record["text"], _verdict_of(record.get("metadata") or {}))
            )
    return queries


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            if not line.strip():
                continue
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            qrels[query_id][doc_id] = int(score)
    return dict(qrels)


def load(root: Path, split: str = "test", *, check_counts: bool = True) -> Dataset:
    documents = _read_documents(root / "corpus.jsonl")
    queries = _read_queries(root / "queries.jsonl")
    qrels = _read_qrels(root / "qrels" / f"{split}.tsv")

    if check_counts:
        if len(documents) != EXPECTED_DOCS:
            raise DatasetError(f"expected {EXPECTED_DOCS} documents, found {len(documents)}")
        if len(queries) != EXPECTED_QUERIES:
            raise DatasetError(f"expected {EXPECTED_QUERIES} queries, found {len(queries)}")

    return Dataset(documents=documents, queries=queries, qrels=qrels)
