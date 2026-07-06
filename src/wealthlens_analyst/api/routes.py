"""API routes.

- POST /ask           — the analyst. Body: {"question": ...}.
                        * Plain mode (H1-20): hybrid retrieval -> cited
                          composition -> citation resolution, returning the
                          published response schema (api/schemas.py): an
                          ``answer`` with resolved citations, or a ``refusal``.
                          Refusal triggers: the abstention gate (H1-21/H1-22)
                          rejects weak/thin fused evidence BEFORE generation (zero
                          generation spend; gate signal logged); or, post-generation,
                          the answer is not fully cited (no citation resolves, or
                          ANY cited id was pruned). The budget guard (H1-27) adds
                          the ``over_budget`` (429) variant. Every request that runs
                          writes a query_log row.
                        * ?debug=retrieval (H1-13) returns the RRF-fused
                          candidate list with per-chunk provenance and both
                          component ranks, WITHOUT generation — a diagnostic
                          surface, not a product answer.
- GET  /healthz       — liveness: app up, DB reachable.
- GET  /metrics/data  — JSON for the public metrics page: p50/p95 latency and
                        cost per query, computed from query_log (ADR 0002).
                        Cache hit rate is added only when caching lands.

Accounting (ADR 0002): every /ask request writes one query_log row carrying the
REAL per-request spend summed across every model call it made (the query
embedding, plus generation in plain mode). A served/refused outcome whose row
cannot be written fails the request (500) rather than serve unmetered spend;
failure paths write an `error` row best-effort without masking the original
error. tokens_out and generation cost are 0 in debug mode (no generation).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Annotated, Literal, Protocol

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from wealthlens_analyst.answer.abstain import evaluate_evidence
from wealthlens_analyst.answer.citations import SourceMeta, resolve_citations
from wealthlens_analyst.answer.compose import (
    ComposedAnswer,
    EmptyGenerationError,
    compose_answer,
    strip_unresolved_citation_markers,
)
from wealthlens_analyst.api.schemas import AnswerResponse, CitationModel, RefusalResponse
from wealthlens_analyst.budget.models import QueryDecision, record_query
from wealthlens_analyst.llm.client import EmbeddingResult, get_client
from wealthlens_analyst.retrieval.dense import search_dense_by_vector
from wealthlens_analyst.retrieval.fts import ChunkHit, search_fts
from wealthlens_analyst.retrieval.fuse_rrf import fuse_rrf

logger = logging.getLogger(__name__)

router = APIRouter()

#: How many candidates each retriever contributes to fusion. Matches the
#: search_fts/search_dense default: deep enough that RRF sees genuine overlap,
#: cheap enough that a single /ask stays one embedding call + two indexed reads.
_PER_RETRIEVER_LIMIT = 50

#: How many fused candidates the debug surface returns. Matches the fuse_rrf
#: default (ADR 0001); H1-14's recall report is measured against this top-N.
_FUSED_LIMIT = 20

#: How many top fused chunks plain mode shows the generator. Bounds generation
#: input tokens (cost) and focuses the model on the strongest evidence, while
#: staying well inside the fused top-N. The abstention gate (H1-21) will decide
#: WHETHER to generate; this only decides how much evidence to pass once we do.
_COMPOSE_EVIDENCE_LIMIT = 8

#: Refusal reasons served in the ``refusal`` variant. Human-readable and honest;
#: both are the "cannot answer from this corpus" outcome (ADR 0003 D4).
_NO_EVIDENCE_REASON = "No matching evidence was found in the corpus for this question."
#: Covers both a model refusal (nothing citable) and a partial prune (not FULLY
#: citable) — in either case a fully cited answer cannot be returned.
_UNSUPPORTED_REASON = "The available evidence does not support a fully cited answer to this question."


class _GenerationSpend(Protocol):
    """The accounting fields shared by ComposedAnswer and CompletionResult.

    Lets the recording helpers take either (a successful answer, or the result
    an EmptyGenerationError carries) without caring which — both report the real
    tokens and cost the request must account for (ADR 0002). Read-only
    properties so the frozen ComposedAnswer/CompletionResult dataclasses satisfy
    it (a plain attribute would demand a settable field).
    """

    @property
    def tokens_in(self) -> int: ...

    @property
    def tokens_out(self) -> int: ...

    @property
    def cost_gbp(self) -> float: ...


def _elapsed_ms(started: float) -> int:
    """Milliseconds since a time.perf_counter() start, rounded to int."""
    return round((time.perf_counter() - started) * 1000)


def _combined_spend(embedded: EmbeddingResult | None, generation: _GenerationSpend | None) -> tuple[int, int, float]:
    """Sum the per-request spend across the embedding and generation calls.

    A query_log row records ONE request's totals: tokens_in/cost are summed over
    every model call the request made (embedding + generation), tokens_out is
    the generation's output (embeddings have none). Either leg may be absent (a
    provider failure before the embed; a refusal before generation).
    """
    tokens_in = (embedded.tokens_in if embedded is not None else 0) + (
        generation.tokens_in if generation is not None else 0
    )
    tokens_out = generation.tokens_out if generation is not None else 0
    cost_gbp = (embedded.cost_gbp if embedded is not None else 0.0) + (
        generation.cost_gbp if generation is not None else 0.0
    )
    return tokens_in, tokens_out, cost_gbp


def _record_or_500(
    engine: Engine,
    *,
    question: str,
    decision: QueryDecision,
    embedded: EmbeddingResult | None,
    generation: _GenerationSpend | None,
    started: float,
    gate_signal: float | None = None,
) -> None:
    """Record a SERVED outcome's row; a write failure fails the request (500).

    Accounting is part of serving the request (ADR 0002): if the row cannot be
    written, fail rather than serve unmetered spend — a silently dropped row
    would systematically understate the summed cap input the budget guard
    (H1-27) enforces against. Used for answered and (served) refused outcomes.
    ``gate_signal`` is the abstention gate's signal (H1-22) when the gate ran.
    """
    tokens_in, tokens_out, cost_gbp = _combined_spend(embedded, generation)
    try:
        record_query(
            engine,
            question=question,
            decision=decision,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_gbp=cost_gbp,
            latency_ms=_elapsed_ms(started),
            gate_signal=gate_signal,
        )
    except SQLAlchemyError as exc:
        logger.error("ask: query_log write failed after a served %s outcome: %s", decision.value, exc)
        raise HTTPException(status_code=500, detail="query accounting failed") from exc


def _record_error_best_effort(
    engine: Engine,
    question: str,
    embedded: EmbeddingResult | None,
    started: float,
    *,
    generation: _GenerationSpend | None = None,
    gate_signal: float | None = None,
) -> None:
    """Record an `error` query_log row on a FAILURE path without masking the failure.

    The original error is what the caller must see; if the accounting write also
    fails (e.g. the same DB outage that broke retrieval), that is logged loudly
    and swallowed — the opposite policy from the served path (see _record_or_500).
    Records whatever WAS spent: the embedding, plus a generation result when the
    failure happened after generation (e.g. a resolution read that failed once
    the answer was already generated and paid for). ``gate_signal`` is recorded
    when the failure happened after the gate ran (H1-22).
    """
    tokens_in, tokens_out, cost_gbp = _combined_spend(embedded, generation)
    try:
        record_query(
            engine,
            question=question,
            decision=QueryDecision.ERROR,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_gbp=cost_gbp,
            latency_ms=_elapsed_ms(started),
            gate_signal=gate_signal,
        )
    except Exception:  # deliberately broad: never mask the original error
        logger.exception("ask: could not record error query_log row")


def get_engine(request: Request) -> Engine:
    """Yield the app's shared engine (created in app.py's lifespan).

    A FastAPI dependency so tests can override it (dependency_overrides)
    without a database; the request path never builds an engine per query.
    """
    engine: Engine = request.app.state.engine
    return engine


def get_registry(request: Request) -> Mapping[str, SourceMeta]:
    """Yield the source registry loaded once at startup (app.py's lifespan).

    A FastAPI dependency (like get_engine) so plain mode resolves a citation's
    source name/URL from an in-memory map, never re-reading registries/sources.yml
    per request; tests override it without touching disk.
    """
    registry: Mapping[str, SourceMeta] = request.app.state.registry
    return registry


class AskRequest(BaseModel):
    """A question for the analyst."""

    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject blank and NUL-bearing questions before they reach retrieval.

        min_length=1 alone would admit " ", which FTS-matches nothing but
        would still trigger a paid embedding call in the dense leg. A NUL
        byte is valid JSON/Python but not valid Postgres text — it would
        surface as a DB error (500) instead of the honest 422.
        """
        if not value.strip():
            raise ValueError("question must not be blank")
        if "\x00" in value:
            raise ValueError("question must not contain NUL characters")
        return value


class RetrievalCandidate(BaseModel):
    """One fused retrieval candidate with its ingestion-time provenance.

    Provenance fields mirror the chunks table columns (ADR 0001 §4) — carried
    through ChunkHit, never reconstructed. fts_rank/dense_rank are the chunk's
    1-based rank within each retriever's own list (None = that retriever did
    not return it), so H1-14's recall analysis can attribute every hit.
    """

    chunk_id: int
    source_id: str
    document_id: str
    section: str | None
    page: int | None
    span: str | None
    text: str
    fused_rank: int
    rrf_score: float
    fts_rank: int | None
    dense_rank: int | None


class RetrievalDebugResponse(BaseModel):
    """The /ask?debug=retrieval payload: fused candidates, no generation.

    A diagnostic surface (mode="retrieval"), deliberately separate from the
    published product-answer schema (api/schemas.py's answer|refusal|over_budget).
    """

    question: str
    mode: Literal["retrieval"] = "retrieval"
    fts_candidates: int
    dense_candidates: int
    candidates: list[RetrievalCandidate]


@router.get("/healthz")
def healthz(engine: Annotated[Engine, Depends(get_engine)]) -> dict[str, str]:
    """Liveness probe: app up AND the database is reachable.

    Database failures surface as SQLAlchemyError (SQLAlchemy wraps every
    DBAPI/psycopg error, including pool timeouts) and map to 503 — the
    probe's honest signal, not a 500: a health endpoint that crashes on the
    condition it exists to detect would be reporting its own bug, not the
    service's state. Anything outside that hierarchy would be a genuine app
    bug, and a 500 is then the correct signal.
    """
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.warning("healthz: database unreachable: %s", exc)
        raise HTTPException(status_code=503, detail="database unreachable") from exc
    return {"status": "ok", "database": "ok"}


def _run_retrieval(
    question: str, engine: Engine, started: float
) -> tuple[list[ChunkHit], list[ChunkHit], EmbeddingResult]:
    """Run both retriever legs and return (lexical, dense, embedded), or raise.

    The legs run sequentially, FREE leg first, on purpose: if the database is
    down, search_fts fails before the query embedding can spend. (Concurrent
    legs are a latency optimisation to revisit once the budget guard, H1-27,
    makes an always-spend race acceptable.) The embed happens HERE, through the
    client seam, rather than inside search_dense, so its accounting can be
    persisted to query_log. On failure an `error` row is recorded (with whatever
    WAS spent) and the mapped HTTPException is raised: a database failure is the
    backend's fault -> 503 (matching /healthz); a provider (embedding) failure
    stays 500. Shared by both the debug and plain surfaces.
    """
    embedded: EmbeddingResult | None = None
    try:
        lexical = search_fts(question, limit=_PER_RETRIEVER_LIMIT, engine=engine)
        embedded = get_client().embed([question])
        dense = search_dense_by_vector(embedded.vectors[0], limit=_PER_RETRIEVER_LIMIT, engine=engine)
    except SQLAlchemyError as exc:
        logger.warning("ask: retrieval backend unavailable: %s", exc)
        # A connection-level failure (OperationalError) dooms the accounting
        # write too — same database — so attempting it would only add a second
        # full connect-timeout cycle to the caller's wait for the 503.
        # Statement-level failures keep the connection alive and still record.
        if isinstance(exc, OperationalError):
            logger.warning("ask: skipping doomed error-row write on a connection-level failure")
        else:
            _record_error_best_effort(engine, question, embedded, started)
        raise HTTPException(status_code=503, detail="retrieval backend unavailable") from exc
    except Exception:
        _record_error_best_effort(engine, question, embedded, started)
        raise
    return lexical, dense, embedded


def _ask_debug_retrieval(question: str, engine: Engine, started: float) -> RetrievalDebugResponse:
    """The ?debug=retrieval surface (H1-13): fused candidates, no generation."""
    lexical, dense, embedded = _run_retrieval(question, engine, started)
    fused = fuse_rrf(lexical, dense, limit=_FUSED_LIMIT)
    # tokens_out/generation cost are 0: debug mode runs no generation.
    _record_or_500(
        engine,
        question=question,
        decision=QueryDecision.ANSWERED,
        embedded=embedded,
        generation=None,
        started=started,
    )
    # ChunkHit.rank is the 1-based position each retriever assigned (fts.py's
    # _hits_from_rows); fuse_rrf overwrites rank/score on ITS copies but the
    # raw input lists still hold the component ranks the response surfaces.
    fts_ranks = {hit.chunk_id: hit.rank for hit in lexical}
    dense_ranks = {hit.chunk_id: hit.rank for hit in dense}
    return RetrievalDebugResponse(
        question=question,
        fts_candidates=len(lexical),
        dense_candidates=len(dense),
        candidates=[
            RetrievalCandidate(
                chunk_id=hit.chunk_id,
                source_id=hit.source_id,
                document_id=hit.document_id,
                section=hit.section,
                page=hit.page,
                span=hit.span,
                text=hit.text,
                fused_rank=hit.rank,
                rrf_score=hit.score,
                fts_rank=fts_ranks.get(hit.chunk_id),
                dense_rank=dense_ranks.get(hit.chunk_id),
            )
            for hit in fused
        ],
    )


def _ask_plain(
    question: str, engine: Engine, registry: Mapping[str, SourceMeta], started: float
) -> AnswerResponse | RefusalResponse:
    """Plain mode (H1-20): retrieve -> compose -> resolve citations -> serve.

    Returns a cited ``answer`` or an honest ``refusal``. Composition and
    resolution run inside an accounted try-block so a generation that spends
    then fails (empty/truncated output, or a resolution read that fails after
    the paid call) still records its spend before surfacing the error.
    """
    lexical, dense, embedded = _run_retrieval(question, engine, started)
    fused = fuse_rrf(lexical, dense, limit=_FUSED_LIMIT)
    # Abstention gate (H1-21) BEFORE generation (H1-22): weak or thin fused
    # evidence refuses here, so the refusal costs ZERO generation spend (ADR 0003
    # D4). This subsumes the empty-fusion case (no evidence -> signal 0.0 -> not
    # answerable). The gate signal is logged to query_log.gate_signal on every
    # row from here on, so refusal behaviour is auditable / calibratable.
    gate = evaluate_evidence(question, fused)
    if not gate.answerable:
        _record_or_500(
            engine,
            question=question,
            decision=QueryDecision.REFUSED,
            embedded=embedded,
            generation=None,  # no generation ran -> tokens_out/gen cost 0
            started=started,
            gate_signal=gate.signal,
        )
        return RefusalResponse(mode="refusal", question=question, reason=_NO_EVIDENCE_REASON)

    composed: ComposedAnswer | None = None
    try:
        composed = compose_answer(question, fused[:_COMPOSE_EVIDENCE_LIMIT])
        resolved = resolve_citations(composed, engine=engine, registry=registry)
    except EmptyGenerationError as exc:
        # Empty or truncated generation: real spend, unusable text. Record the
        # embed + generation spend (the exception carries the CompletionResult)
        # then fail — a possibly cut-off figure must never be served.
        _record_error_best_effort(engine, question, embedded, started, generation=exc.result, gate_signal=gate.signal)
        raise HTTPException(status_code=500, detail="answer generation failed") from exc
    except SQLAlchemyError as exc:
        # Only the resolution read can raise this here (retrieval already
        # returned). Unlike _run_retrieval — which skips the write on a
        # connection-level (OperationalError) failure because at most one
        # embedding's spend is at stake — a full generation is ALWAYS already
        # paid by this point (the dominant cost). So its spend is recorded even
        # on a connection-level failure, accepting one extra connect-timeout on
        # the 503: the realistic trigger is a transient blip during the
        # multi-second generation window, where the DB is often reachable again
        # by the time record_query runs, so the best-effort write usually lands.
        logger.warning("ask: citation resolution backend unavailable: %s", exc)
        _record_error_best_effort(engine, question, embedded, started, generation=composed, gate_signal=gate.signal)
        raise HTTPException(status_code=503, detail="retrieval backend unavailable") from exc
    except Exception:
        _record_error_best_effort(engine, question, embedded, started, generation=composed, gate_signal=gate.signal)
        raise

    if not resolved.citations or resolved.unresolved_chunk_ids:
        # Refuse unless the answer is FULLY cited. Two triggers: the model cited
        # nothing that resolves (its refusal sentence), OR at least one cited id
        # was pruned (fabricated / missing / unknown-source). Serving a partially
        # pruned answer would leave its now-uncited claim in the body — exactly
        # what a citation-first product must not do. The generation DID spend, so
        # the refused row carries that spend.
        _record_or_500(
            engine,
            question=question,
            decision=QueryDecision.REFUSED,
            embedded=embedded,
            generation=composed,
            started=started,
            gate_signal=gate.signal,
        )
        return RefusalResponse(mode="refusal", question=question, reason=_UNSUPPORTED_REASON)

    # Every cited id resolved. Strip any drift / out-of-range citation-shaped text
    # the strict parser never counted, so only canonical served-citation markers
    # remain in the body, then serve.
    resolved_ids = {citation.chunk_id for citation in resolved.citations}
    served_text = strip_unresolved_citation_markers(composed.text, resolved_ids).strip()
    _record_or_500(
        engine,
        question=question,
        decision=QueryDecision.ANSWERED,
        embedded=embedded,
        generation=composed,
        started=started,
        gate_signal=gate.signal,
    )
    return AnswerResponse(
        mode="answer",
        question=question,
        answer=served_text,
        citations=[CitationModel.from_citation(citation) for citation in resolved.citations],
    )


@router.post("/ask")
def ask(
    request: AskRequest,
    engine: Annotated[Engine, Depends(get_engine)],
    registry: Annotated[Mapping[str, SourceMeta], Depends(get_registry)],
    debug: Literal["retrieval"] | None = None,
) -> AnswerResponse | RefusalResponse | RetrievalDebugResponse:
    """Answer a question with chunk-level citations, refuse honestly, or diagnose.

    Plain mode (H1-20) returns the published response schema (api/schemas.py):
    an ``answer`` with resolved citations, or a ``refusal`` when the corpus does
    not support a cited answer. ?debug=retrieval (H1-13) returns the RRF-fused
    candidate list without generation. Requests rejected by validation (422)
    never reach the handler and log nothing; every request that DOES run writes
    exactly one query_log row (see the module docstring's accounting note).

    An unknown debug value is a 422 (the Literal), not a silent plain-mode
    fallthrough. The abstention gate (H1-21/H1-22) refuses weak evidence ahead of
    generation; the budget guard (H1-27) will add the over_budget (429) variant
    ahead of retrieval.
    """
    started = time.perf_counter()
    if debug == "retrieval":
        return _ask_debug_retrieval(request.question, engine, started)
    return _ask_plain(request.question, engine, registry, started)


@router.get("/metrics/data")
def metrics_data() -> dict[str, object]:
    """Serve p50/p95 latency and cost-per-query aggregates from query_log."""
    raise NotImplementedError("H1-29: metrics endpoint not yet implemented")
