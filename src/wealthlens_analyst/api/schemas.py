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

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from wealthlens_analyst.answer.citations import Citation
from wealthlens_analyst.budget.models import QueryDecision


class MetricsResponse(BaseModel):
    """Public /metrics/data JSON for the current UTC calendar month.

    The window is [window_start, window_end), matching the budget spend month.
    Latency uses answered+refused rows and nearest-rank p50/p95 (ceil(p*n)).
    Costs include all decisions, including zero-cost rows and billed errors;
    an even cost median averages the two middle values. Decimal costs serialize
    as strings with eight places (ROUND_HALF_EVEN); empty samples are null.
    Counts describe logged /ask requests, not all HTTP traffic (e.g. no 422s).
    """

    window_start: datetime = Field(description="Inclusive UTC calendar-month start.")
    window_end: datetime = Field(description="Exclusive UTC calendar-month end.")
    total_queries: int = Field(description="All query_log rows in the window; the cost/query denominator.")
    decision_counts: dict[QueryDecision, int] = Field(description="Counts for all four decisions, including zeros.")
    latency_sample_count: int = Field(description="Number of answered + refused rows used for latency.")
    latency_p50_ms: int | None = Field(description="Nearest-rank 50th percentile latency in milliseconds.")
    latency_p95_ms: int | None = Field(description="Nearest-rank 95th percentile latency in milliseconds.")
    cost_mean_gbp: Decimal | None = Field(description="Mean GBP per logged query, string rounded to eight places.")
    cost_median_gbp: Decimal | None = Field(description="Median GBP per logged query, string rounded to eight places.")


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
