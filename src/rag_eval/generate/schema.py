"""What the verifier is asked to produce."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_eval.data.scifact import CONTRADICT, NOINFO, SUPPORT

PROMPT_VERSION = "v1"


class Verification(BaseModel):
    """A verdict on a claim, with the passages that carry it.

    Structured output rather than free text: a parse failure would land in the
    metrics as a wrong answer, which would measure the parser instead of the
    model.
    """

    verdict: str = Field(description=f"one of {SUPPORT}, {CONTRADICT}, {NOINFO}")
    rationale: str = Field(description="one or two sentences, grounded in the passages")
    cited_indices: list[int] = Field(
        default_factory=list, description="1-based indices of the passages used"
    )


class JudgeVerdict(BaseModel):
    """The judge's own reading of the same claim, used to measure the judge."""

    verdict: str = Field(description=f"one of {SUPPORT}, {CONTRADICT}, {NOINFO}")
    grounded: bool = Field(description="whether the cited passages actually carry the verdict")
    note: str = Field(default="", description="one short sentence")
