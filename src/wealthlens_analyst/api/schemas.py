"""Published /ask response schema — the product answer contract (H1-20).

Three variants, discriminated on ``mode``:

- ``answer``      — a cited answer (HTTP 200).
- ``refusal``     — honest "cannot answer from this corpus" (HTTP 200). A
                    refusal is a product feature, not an error path (ADR 0003 D4).
- ``over_budget`` — the monthly spend cap was exceeded (HTTP 429). The variant is
                    published here so the contract is stable; it is *produced* by
                    the budget guard (H1-27), which is not wired yet.

The ``?debug=retrieval`` diagnostic surface (``routes.RetrievalDebugResponse``)
is deliberately NOT part of this product schema — it is a developer surface, not
an answer a user consumes, so it must not widen the published contract.

This module is the single source of truth for the contract. The committed
``evals/checks/ask_response.schema.json`` is generated from it (drift-locked by a
test), and the deterministic live check validates real /ask responses against
that file — the H1-20 done-when.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from wealthlens_analyst.answer.citations import Citation


class CitationModel(BaseModel):
    """A resolved, render-ready citation (the wire form of answer.citations.Citation).

    Carries the full ADR 0001 §4 provenance tuple (section/page/span) plus the
    registry source name + URL and the ingestion access date, so every served
    citation carries both a URL and an access date (the data-integrity rule).
    """

    chunk_id: int
    source_id: str
    source_name: str
    document_id: str
    section: str | None
    page: int | None
    span: str | None
    url: str
    access_date: date

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationModel:
        """Project a resolved ``Citation`` onto its wire model."""
        return cls(
            chunk_id=citation.chunk_id,
            source_id=citation.source_id,
            source_name=citation.source_name,
            document_id=citation.document_id,
            section=citation.section,
            page=citation.page,
            span=citation.span,
            url=citation.url,
            access_date=citation.access_date,
        )


class AnswerResponse(BaseModel):
    """A cited answer to an in-corpus question.

    An answer is served ONLY when EVERY cited id resolved — "answer" means a
    fully-cited answer. The serving path refuses if any citation was pruned
    (fabricated / missing / unknown-source), so the body never carries a claim
    whose citation could not be verified, and ``citations`` is always non-empty
    (``minItems: 1`` in the published schema). ``answer`` keeps only canonical
    ``[chunk:<id>]`` markers for served citations; any drift / out-of-range
    citation-shaped text the model emitted is stripped before serving.

    ``mode`` has no default so the published schema marks the discriminator
    ``required`` — a body without it is not a valid response.
    """

    mode: Literal["answer"]
    question: str
    answer: str
    citations: list[CitationModel] = Field(min_length=1)


class RefusalResponse(BaseModel):
    """Honest abstention: the corpus does not support a cited answer (ADR 0003 D4)."""

    mode: Literal["refusal"]
    question: str
    reason: str


class OverBudgetResponse(BaseModel):
    """The monthly spend cap was exceeded (HTTP 429). Produced by the guard (H1-27)."""

    mode: Literal["over_budget"]
    question: str
    reason: str


#: The published product-answer union, discriminated on ``mode``. A response is
#: exactly one of the three variants; the discriminator makes serialisation and
#: validation unambiguous.
AskResponse = Annotated[
    AnswerResponse | RefusalResponse | OverBudgetResponse,
    Field(discriminator="mode"),
]

_ASK_RESPONSE_ADAPTER: TypeAdapter[Any] = TypeAdapter(AskResponse)


def ask_response_json_schema() -> dict[str, Any]:
    """Return the published JSON Schema for the /ask product-response union.

    Generated from the Pydantic models above so the committed schema file can
    never silently drift from the code (a test regenerates and compares).
    """
    return _ASK_RESPONSE_ADAPTER.json_schema()
