"""Verifying a claim against the passages that were retrieved.

The verifier only ever sees the passages. If it could search the corpus, the
number would measure an agent rather than the retrieval it sits on.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.data.scifact import CONTRADICT, NOINFO, SUPPORT, VERDICTS, Document
from rag_eval.generate.provider import LlmProvider, Usage
from rag_eval.generate.schema import PROMPT_VERSION, Verification

SYSTEM = (
    "You check scientific claims against passages from paper abstracts.\n"
    f"Answer {SUPPORT} if the passages support the claim, {CONTRADICT} if they contradict it, "
    f"and {NOINFO} if they do not settle it.\n"
    f"{NOINFO} is a correct answer, not a failure: say it rather than reaching.\n"
    "Cite the passages you used by their number. Do not use outside knowledge."
)


@dataclass(frozen=True)
class Verified:
    query_id: str
    verdict: str
    rationale: str
    cited_indices: list[int]
    retrieved_ids: list[str]
    faithful_citations: bool
    usage: Usage
    prompt_version: str = PROMPT_VERSION

    @property
    def cited_doc_ids(self) -> list[str]:
        return [
            self.retrieved_ids[i - 1]
            for i in self.cited_indices
            if 1 <= i <= len(self.retrieved_ids)
        ]


def build_prompt(claim: str, passages: list[Document]) -> str:
    numbered = "\n\n".join(
        f"[{position}] {doc.title}\n{doc.text}" for position, doc in enumerate(passages, start=1)
    )
    return f"Claim: {claim}\n\nPassages:\n{numbered}"


def normalize_verdict(raw: str) -> str:
    """Map a verdict onto the label set, refusing anything else.

    Silently turning an unknown string into NOINFO would make the accuracy
    number describe a fallback rather than the model.
    """
    candidate = raw.strip().upper().replace("NO_INFO", NOINFO).replace("NOT ENOUGH INFO", NOINFO)
    if candidate not in VERDICTS:
        raise ValueError(f"verdict outside the label set: {raw!r}")
    return candidate


def verify(
    provider: LlmProvider,
    query_id: str,
    claim: str,
    passages: list[Document],
) -> Verified:
    parsed, usage = provider.complete(SYSTEM, build_prompt(claim, passages), Verification)
    retrieved_ids = [doc.doc_id for doc in passages]

    # A citation outside the range is counted, not discarded. Dropping it would
    # erase the very defect the faithfulness metric exists to show.
    faithful = all(1 <= index <= len(passages) for index in parsed.cited_indices)

    return Verified(
        query_id=query_id,
        verdict=normalize_verdict(parsed.verdict),
        rationale=parsed.rationale,
        cited_indices=list(parsed.cited_indices),
        retrieved_ids=retrieved_ids,
        faithful_citations=faithful,
        usage=usage,
    )
