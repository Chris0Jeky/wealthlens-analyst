"""Unit tests for the /ask surfaces (plain H1-20 + ?debug=retrieval H1-13) and /healthz.

The retrieval SQL and generation both need real backends, so the retriever legs,
the client seam, compose_answer and resolve_citations are monkeypatched here with
fixtures and verified live against analyst-db separately (the backlog's
done-when). These tests pin what must hold WITHOUT a database or a model call:
the fusion wiring, the component-rank attribution, the provenance passthrough,
the plain-mode answer/refusal contract + serving policy (pruned markers stripped,
spend summed across embed + generation), and the failure contracts (422 bad
input, 500 generation/accounting failure, 503 unreachable DB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.exc import DatabaseError, OperationalError

import wealthlens_analyst.api.routes as routes
from wealthlens_analyst.answer.abstain import GateDecision, evaluate_evidence
from wealthlens_analyst.answer.citations import Citation, ResolvedCitations
from wealthlens_analyst.answer.compose import ComposedAnswer, EmptyGenerationError
from wealthlens_analyst.api.app import create_app
from wealthlens_analyst.api.routes import get_engine, get_registry
from wealthlens_analyst.budget.models import QueryDecision
from wealthlens_analyst.llm.client import CompletionResult, EmbeddingResult
from wealthlens_analyst.retrieval.fts import ChunkHit
from wealthlens_analyst.retrieval.fuse_rrf import RRF_K


def _hit(chunk_id: int, rank: int, score: float) -> ChunkHit:
    """A ChunkHit with distinguishable provenance derived from chunk_id."""
    return ChunkHit(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        document_id=f"doc-{chunk_id}",
        section=f"section-{chunk_id}",
        page=None,
        span=f"span-{chunk_id}",
        text=f"text of chunk {chunk_id}",
        rank=rank,
        score=score,
    )


# Overlap fixture: chunk 2 appears in BOTH lists (top of dense, second in
# lexical) so fusion must rank it first and report both component ranks.
_LEXICAL = [_hit(1, rank=1, score=0.9), _hit(2, rank=2, score=0.5), _hit(3, rank=3, score=0.1)]
_DENSE = [_hit(2, rank=1, score=0.8), _hit(4, rank=2, score=0.7)]

_ENGINE_SENTINEL = object()
#: The registry the route injects into resolve_citations (H1-20). Its contents
#: are irrelevant to the DB-free tests (resolve_citations is monkeypatched), so
#: a sentinel proves only that the wiring passes the shared object through.
_REGISTRY_SENTINEL: dict[str, Any] = {}

# What the fake client's embed() reports — the accounting the query_log row
# must carry (H1-15).
_EMBED_TOKENS = 7
_EMBED_COST_GBP = 2.5e-7

# What a monkeypatched generation reports (plain mode, H1-20) — the query_log
# row must carry embed + generation spend summed.
_GEN_TOKENS_IN = 120
_GEN_TOKENS_OUT = 30
_GEN_COST_GBP = 1.5e-3

# The harness makes the abstention gate (H1-22) ANSWERABLE by default with this
# signal, so compose-focused tests exercise the compose/resolve/serve seam in
# isolation (not coupled to the gate's uncalibrated thresholds) AND every
# post-gate query_log row carries a known gate_signal to assert. Gate-focused
# tests override routes.evaluate_evidence themselves.
_DEFAULT_GATE_SIGNAL = 0.5


def _composed(text: str, cited: list[int], *, fabricated: list[int] | None = None) -> ComposedAnswer:
    """A ComposedAnswer with the standard test generation accounting."""
    return ComposedAnswer(
        text=text,
        cited_chunk_ids=cited,
        tokens_in=_GEN_TOKENS_IN,
        tokens_out=_GEN_TOKENS_OUT,
        cost_gbp=_GEN_COST_GBP,
        model="fake-analyst-model",
        fabricated_chunk_ids=fabricated or [],
    )


def _citation(chunk_id: int) -> Citation:
    """A fully-resolved Citation with distinguishable provenance."""
    return Citation(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        source_name=f"Source {chunk_id}",
        document_id=f"doc-{chunk_id}",
        section=f"section-{chunk_id}",
        page=None,
        span=f"span-{chunk_id}",
        url=f"https://example.org/{chunk_id}",
        access_date=date(2026, 6, 1),
    )


class _FakeClient:
    """Stands in for the llm/client seam: one canned embedding, no network."""

    def __init__(self, calls: dict[str, dict[str, Any]]) -> None:
        self._calls = calls

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self._calls["embed"] = {"texts": list(texts)}
        return EmbeddingResult(
            vectors=[[0.1, 0.2, 0.3]],
            model="fake-embedding-model",
            tokens_in=_EMBED_TOKENS,
            cost_gbp=_EMBED_COST_GBP,
        )


@dataclass
class _Harness:
    """A DB-free app: fixture retriever legs, sentinel engine, recorded calls."""

    client: TestClient
    app: FastAPI
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    recorded: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """Build the app with monkeypatched retrievers and an overridden engine.

    The TestClient is used WITHOUT entering its context manager, so the
    lifespan (real engine creation) never runs — these tests are DB-free.
    record_query is captured (not written): its row shape is pinned in
    test_budget_models.py and the real insert is verified live.
    """
    calls: dict[str, dict[str, Any]] = {}
    recorded: list[dict[str, Any]] = []

    def fake_fts(query: str, *, limit: int, engine: Any) -> list[ChunkHit]:
        calls["fts"] = {"query": query, "limit": limit, "engine": engine}
        return list(_LEXICAL)

    def fake_dense_by_vector(query_vec: list[float], *, limit: int, engine: Any) -> list[ChunkHit]:
        calls["dense"] = {"query_vec": query_vec, "limit": limit, "engine": engine}
        return list(_DENSE)

    def fake_record_query(engine: Any, **kwargs: Any) -> None:
        recorded.append({"engine": engine, **kwargs})

    monkeypatch.setattr(routes, "search_fts", fake_fts)
    monkeypatch.setattr(routes, "search_dense_by_vector", fake_dense_by_vector)
    monkeypatch.setattr(routes, "get_client", lambda: _FakeClient(calls))
    monkeypatch.setattr(routes, "record_query", fake_record_query)
    # Default: gate answerable (compose-focused tests are decoupled from the
    # gate's uncalibrated thresholds). Gate-focused tests override this.
    monkeypatch.setattr(
        routes,
        "evaluate_evidence",
        lambda q, ev: GateDecision(answerable=True, signal=_DEFAULT_GATE_SIGNAL, reason=None),
    )
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: _ENGINE_SENTINEL
    # get_registry reads app.state.registry (set at startup); the lifespan never
    # runs in these DB-free tests, so inject the sentinel the same way as engine.
    app.dependency_overrides[get_registry] = lambda: _REGISTRY_SENTINEL
    return _Harness(client=TestClient(app), app=app, calls=calls, recorded=recorded)


def test_ask_debug_retrieval_fuses_and_attributes_component_ranks(harness: _Harness) -> None:
    response = harness.client.post("/ask?debug=retrieval", json={"question": "top decile wealth"})
    assert response.status_code == 200
    body = response.json()

    assert body["question"] == "top decile wealth"
    assert body["mode"] == "retrieval"
    assert body["fts_candidates"] == 3
    assert body["dense_candidates"] == 2

    # RRF (k=60): chunk 2 is in both lists so it must fuse first; the rest
    # follow by their single-list contribution 1/(k + position).
    assert [c["chunk_id"] for c in body["candidates"]] == [2, 1, 4, 3]
    assert [c["fused_rank"] for c in body["candidates"]] == [1, 2, 3, 4]

    by_id = {c["chunk_id"]: c for c in body["candidates"]}
    assert by_id[2]["rrf_score"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert by_id[1]["rrf_score"] == pytest.approx(1 / (RRF_K + 1))
    assert by_id[4]["rrf_score"] == pytest.approx(1 / (RRF_K + 2))
    assert by_id[3]["rrf_score"] == pytest.approx(1 / (RRF_K + 3))

    # Component-rank attribution: both ranks when in both lists, None otherwise.
    assert (by_id[2]["fts_rank"], by_id[2]["dense_rank"]) == (2, 1)
    assert (by_id[1]["fts_rank"], by_id[1]["dense_rank"]) == (1, None)
    assert (by_id[4]["fts_rank"], by_id[4]["dense_rank"]) == (None, 2)
    assert (by_id[3]["fts_rank"], by_id[3]["dense_rank"]) == (3, None)


def test_ask_debug_retrieval_passes_provenance_through_unchanged(harness: _Harness) -> None:
    response = harness.client.post("/ask?debug=retrieval", json={"question": "median wealth"})
    top = response.json()["candidates"][0]  # chunk 2
    assert top["source_id"] == "source-2"
    assert top["document_id"] == "doc-2"
    assert top["section"] == "section-2"
    assert top["page"] is None
    assert top["span"] == "span-2"
    assert top["text"] == "text of chunk 2"


def test_ask_debug_retrieval_wires_shared_engine_into_both_legs(harness: _Harness) -> None:
    harness.client.post("/ask?debug=retrieval", json={"question": "capital gains"})
    assert harness.calls["fts"]["engine"] is _ENGINE_SENTINEL
    assert harness.calls["dense"]["engine"] is _ENGINE_SENTINEL
    assert harness.calls["fts"]["query"] == "capital gains"
    # The route embeds through the seam and passes the vector to the dense leg.
    assert harness.calls["embed"]["texts"] == ["capital gains"]
    assert harness.calls["dense"]["query_vec"] == [0.1, 0.2, 0.3]
    assert harness.calls["fts"]["limit"] == harness.calls["dense"]["limit"] == routes._PER_RETRIEVER_LIMIT


def test_ask_debug_retrieval_records_an_answered_row_with_real_embed_accounting(harness: _Harness) -> None:
    response = harness.client.post("/ask?debug=retrieval", json={"question": "top decile wealth"})
    assert response.status_code == 200
    assert len(harness.recorded) == 1
    row = harness.recorded[0]
    assert row["engine"] is _ENGINE_SENTINEL
    assert row["question"] == "top decile wealth"
    assert row["decision"] == QueryDecision.ANSWERED
    # The backlog's core requirement: debug rows carry the REAL embed spend,
    # not zeros (ADR 0002: every model call is metered).
    assert row["tokens_in"] == _EMBED_TOKENS
    assert row["tokens_out"] == 0
    assert row["cost_gbp"] == _EMBED_COST_GBP
    assert isinstance(row["latency_ms"], int)
    assert row["latency_ms"] >= 0


def test_ask_fails_the_request_when_success_accounting_cannot_be_written(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR 0002 fail-closed: served-but-unrecorded spend would silently
    # understate the summed cap input, so the request must fail instead.
    def broken_record(engine: Any, **kwargs: Any) -> None:
        raise OperationalError("INSERT ...", {}, Exception("db gone"))

    monkeypatch.setattr(routes, "record_query", broken_record)
    response = harness.client.post("/ask?debug=retrieval", json={"question": "anything"})
    assert response.status_code == 500
    assert response.json()["detail"] == "query accounting failed"


def test_ask_plain_returns_a_cited_answer(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # Plain mode (H1-20): retrieve -> compose -> resolve -> serve a cited answer.
    monkeypatch.setattr(routes, "compose_answer", lambda q, ev: _composed("Top decile holds most [chunk:2].", [2]))
    monkeypatch.setattr(
        routes,
        "resolve_citations",
        lambda answer, **kw: ResolvedCitations(citations=[_citation(2)], unresolved_chunk_ids=[]),
    )
    response = harness.client.post("/ask", json={"question": "top decile wealth"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "answer"
    assert body["question"] == "top decile wealth"
    # A resolved marker is preserved inline; nothing was pruned.
    assert body["answer"] == "Top decile holds most [chunk:2]."
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["chunk_id"] == 2
    assert citation["source_name"] == "Source 2"
    assert citation["url"] == "https://example.org/2"
    assert citation["access_date"] == "2026-06-01"


def test_ask_plain_answer_row_sums_embed_and_generation_spend(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes, "compose_answer", lambda q, ev: _composed("A [chunk:2].", [2]))
    monkeypatch.setattr(
        routes,
        "resolve_citations",
        lambda answer, **kw: ResolvedCitations(citations=[_citation(2)], unresolved_chunk_ids=[]),
    )
    harness.client.post("/ask", json={"question": "q"})
    assert len(harness.recorded) == 1
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.ANSWERED
    # Per-request totals across BOTH model calls (H1-20 extends query_log to
    # generation spend): tokens_in summed, tokens_out is the generation's.
    assert row["tokens_in"] == _EMBED_TOKENS + _GEN_TOKENS_IN
    assert row["tokens_out"] == _GEN_TOKENS_OUT
    assert row["cost_gbp"] == pytest.approx(_EMBED_COST_GBP + _GEN_COST_GBP)
    # The gate's signal is logged on the answered row (H1-22).
    assert row["gate_signal"] == _DEFAULT_GATE_SIGNAL


def test_ask_plain_gate_refuses_weak_evidence_before_generation(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The abstention gate (H1-22) refuses BEFORE generation: compose must not
    # run, the refusal costs zero generation spend (embed only), and the gate
    # signal is logged to query_log for auditability/calibration.
    monkeypatch.setattr(
        routes,
        "evaluate_evidence",
        lambda q, ev: GateDecision(answerable=False, signal=0.123, reason="cannot_answer_from_corpus"),
    )

    def _must_not_compose(question: str, evidence: list[ChunkHit]) -> ComposedAnswer:
        raise AssertionError("compose_answer must not run when the gate refuses")

    monkeypatch.setattr(routes, "compose_answer", _must_not_compose)
    response = harness.client.post("/ask", json={"question": "weak evidence"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "refusal"
    assert body["reason"] == routes._NO_EVIDENCE_REASON
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.REFUSED
    assert row["gate_signal"] == 0.123
    assert row["tokens_out"] == 0  # no generation ran
    assert row["tokens_in"] == _EMBED_TOKENS
    assert row["cost_gbp"] == _EMBED_COST_GBP


def test_ask_plain_refuses_on_a_partial_prune_rather_than_serve_an_uncited_claim(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model cited a real chunk (2) AND a fabricated one (99); resolution keeps
    # 2 and prunes 99. Serving the stripped body would leave the [chunk:99] claim
    # uncited — so an answer that is not FULLY cited is refused, not served
    # (the citation-first serving policy). Generation spend is still recorded.
    monkeypatch.setattr(
        routes,
        "compose_answer",
        lambda q, ev: _composed("Real [chunk:2] but also fake [chunk:99].", [2, 99], fabricated=[99]),
    )
    monkeypatch.setattr(
        routes,
        "resolve_citations",
        lambda answer, **kw: ResolvedCitations(citations=[_citation(2)], unresolved_chunk_ids=[99]),
    )
    response = harness.client.post("/ask", json={"question": "q"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "refusal"
    assert body["reason"] == routes._UNSUPPORTED_REASON
    assert harness.recorded[0]["decision"] == QueryDecision.REFUSED
    assert harness.recorded[0]["tokens_out"] == _GEN_TOKENS_OUT  # generation happened
    assert harness.recorded[0]["gate_signal"] == _DEFAULT_GATE_SIGNAL


def test_ask_plain_strips_drift_and_out_of_range_markers_from_a_served_answer(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A FULLY-cited answer (nothing pruned) whose body still carries citation-shaped
    # text the strict parser never counted — a drift form "[chunk: 5]" and a
    # >BIGINT id compose dropped from cited_chunk_ids — must be scrubbed so only the
    # canonical served-citation marker [chunk:2] remains.
    text = "Real [chunk:2], drift [chunk: 5], huge [chunk:99999999999999999999999]."
    monkeypatch.setattr(routes, "compose_answer", lambda q, ev: _composed(text, [2]))
    monkeypatch.setattr(
        routes,
        "resolve_citations",
        lambda answer, **kw: ResolvedCitations(citations=[_citation(2)], unresolved_chunk_ids=[]),
    )
    response = harness.client.post("/ask", json={"question": "q"})
    body = response.json()
    assert body["mode"] == "answer"
    assert body["answer"] == "Real [chunk:2], drift, huge."
    assert [c["chunk_id"] for c in body["citations"]] == [2]


def test_ask_plain_refuses_when_no_citation_resolves(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # The model emitted its mandated refusal sentence (zero citations): a
    # citation-free factual answer must not be served -> honest refusal, with
    # the generation spend still recorded.
    monkeypatch.setattr(
        routes, "compose_answer", lambda q, ev: _composed("The evidence does not support an answer.", [])
    )
    monkeypatch.setattr(routes, "resolve_citations", lambda answer, **kw: ResolvedCitations())
    response = harness.client.post("/ask", json={"question": "unanswerable in corpus"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "refusal"
    assert body["reason"] == routes._UNSUPPORTED_REASON
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.REFUSED
    # Generation happened, so the refused row carries embed + generation spend.
    assert row["tokens_out"] == _GEN_TOKENS_OUT
    assert row["cost_gbp"] == pytest.approx(_EMBED_COST_GBP + _GEN_COST_GBP)
    assert row["gate_signal"] == _DEFAULT_GATE_SIGNAL


def test_ask_plain_refuses_when_every_cited_id_is_pruned(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # The model DID cite (non-empty cited_chunk_ids) but resolution prunes every
    # id (fabricated / missing / unknown-source), so resolved.citations is empty.
    # This is the case the refusal predicate exists for: it distinguishes the
    # correct `not resolved.citations` from the wrong `not composed.cited_chunk_ids`
    # (the latter would SERVE an uncited factual answer — the leak invariant 2
    # forbids). The pruned marker must be stripped even though we refuse.
    monkeypatch.setattr(
        routes, "compose_answer", lambda q, ev: _composed("Gold surged [chunk:99].", [99], fabricated=[99])
    )
    monkeypatch.setattr(
        routes, "resolve_citations", lambda answer, **kw: ResolvedCitations(citations=[], unresolved_chunk_ids=[99])
    )
    response = harness.client.post("/ask", json={"question": "off-corpus but the model answered anyway"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "refusal"
    assert body["reason"] == routes._UNSUPPORTED_REASON
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.REFUSED
    assert row["tokens_out"] == _GEN_TOKENS_OUT  # generation happened before the prune
    assert row["gate_signal"] == _DEFAULT_GATE_SIGNAL


def test_ask_plain_refuses_without_generating_when_no_evidence(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Zero fused candidates: the REAL gate subsumes this (empty -> signal 0.0 ->
    # not answerable) and refuses BEFORE generation. Integration check, so it
    # restores the real gate over the harness's answerable default. The refused
    # row carries embed-only spend (tokens_out=0) and gate_signal 0.0.
    monkeypatch.setattr(routes, "evaluate_evidence", evaluate_evidence)
    monkeypatch.setattr(routes, "search_fts", lambda q, *, limit, engine: [])
    monkeypatch.setattr(routes, "search_dense_by_vector", lambda v, *, limit, engine: [])

    def _must_not_compose(question: str, evidence: list[ChunkHit]) -> ComposedAnswer:
        raise AssertionError("compose_answer must not run when there is no evidence")

    monkeypatch.setattr(routes, "compose_answer", _must_not_compose)
    response = harness.client.post("/ask", json={"question": "nothing matches"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "refusal"
    assert body["reason"] == routes._NO_EVIDENCE_REASON
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.REFUSED
    assert row["tokens_in"] == _EMBED_TOKENS
    assert row["tokens_out"] == 0
    assert row["cost_gbp"] == _EMBED_COST_GBP
    assert row["gate_signal"] == 0.0


def test_ask_plain_wires_shared_engine_and_registry_into_resolution(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_resolve(answer: ComposedAnswer, **kwargs: Any) -> ResolvedCitations:
        captured.update(kwargs)
        return ResolvedCitations(citations=[_citation(2)], unresolved_chunk_ids=[])

    monkeypatch.setattr(routes, "compose_answer", lambda q, ev: _composed("A [chunk:2].", [2]))
    monkeypatch.setattr(routes, "resolve_citations", fake_resolve)
    harness.client.post("/ask", json={"question": "q"})
    # The route passes the SHARED engine and the startup-loaded registry through.
    assert captured["engine"] is _ENGINE_SENTINEL
    assert captured["registry"] is _REGISTRY_SENTINEL


def test_ask_plain_passes_only_the_compose_evidence_limit_to_generation(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 30 disjoint hits per leg -> 20 fused; compose must see only the top
    # _COMPOSE_EVIDENCE_LIMIT of them (bounds generation input).
    lexical = [_hit(i, rank=i, score=1.0 / i) for i in range(1, 31)]
    dense = [_hit(100 + i, rank=i, score=1.0 / i) for i in range(1, 31)]
    monkeypatch.setattr(routes, "search_fts", lambda q, *, limit, engine: lexical)
    monkeypatch.setattr(routes, "search_dense_by_vector", lambda v, *, limit, engine: dense)
    seen: dict[str, int] = {}

    def fake_compose(question: str, evidence: list[ChunkHit]) -> ComposedAnswer:
        seen["count"] = len(evidence)
        return _composed("A [chunk:1].", [1])

    monkeypatch.setattr(routes, "compose_answer", fake_compose)
    monkeypatch.setattr(
        routes,
        "resolve_citations",
        lambda answer, **kw: ResolvedCitations(citations=[_citation(1)], unresolved_chunk_ids=[]),
    )
    harness.client.post("/ask", json={"question": "deep corpus"})
    assert seen["count"] == routes._COMPOSE_EVIDENCE_LIMIT


def test_ask_plain_empty_generation_is_500_and_records_the_spend(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A truncated/empty generation spent real tokens but has no usable text:
    # 500, and the ERROR row must carry embed + generation spend (the exception
    # carries the CompletionResult so the spend is never lost).
    truncated = CompletionResult(
        text="",
        model="fake-analyst-model",
        tokens_in=_GEN_TOKENS_IN,
        tokens_out=900,
        cost_gbp=_GEN_COST_GBP,
        finish_reason="length",
    )

    def boom(question: str, evidence: list[ChunkHit]) -> ComposedAnswer:
        raise EmptyGenerationError("truncated", truncated)

    monkeypatch.setattr(routes, "compose_answer", boom)
    response = harness.client.post("/ask", json={"question": "q"})
    assert response.status_code == 500
    assert response.json()["detail"] == "answer generation failed"
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.ERROR
    assert row["tokens_in"] == _EMBED_TOKENS + _GEN_TOKENS_IN
    assert row["tokens_out"] == 900
    assert row["cost_gbp"] == pytest.approx(_EMBED_COST_GBP + _GEN_COST_GBP)
    assert row["gate_signal"] == _DEFAULT_GATE_SIGNAL  # the gate ran before the failed generation


@pytest.mark.parametrize(
    "exc",
    [
        DatabaseError("SELECT ...", {}, Exception("statement failed")),
        OperationalError("SELECT ...", {}, Exception("connection refused")),
    ],
    ids=["statement-level", "connection-level"],
)
def test_ask_plain_resolution_db_failure_records_generation_spend_then_503(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, exc: DatabaseError
) -> None:
    # resolve_citations' provenance read fails AFTER generation was paid for.
    # A generation is ALWAYS already paid here (unlike the retrieval leg's
    # embed-only skip), so BOTH a statement-level (connection alive) AND a
    # connection-level (OperationalError) failure must record the generation
    # spend in the ERROR row; the caller sees 503.
    monkeypatch.setattr(routes, "compose_answer", lambda q, ev: _composed("A [chunk:2].", [2]))

    def broken_resolve(answer: ComposedAnswer, **kwargs: Any) -> ResolvedCitations:
        raise exc

    monkeypatch.setattr(routes, "resolve_citations", broken_resolve)
    response = harness.client.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert response.json()["detail"] == "retrieval backend unavailable"
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.ERROR
    assert row["tokens_in"] == _EMBED_TOKENS + _GEN_TOKENS_IN
    assert row["tokens_out"] == _GEN_TOKENS_OUT
    assert row["gate_signal"] == _DEFAULT_GATE_SIGNAL  # threaded onto the post-gate error row


def test_ask_plain_fails_the_request_when_answer_accounting_cannot_be_written(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same ADR 0002 fail-closed policy as debug mode, now on the plain path.
    monkeypatch.setattr(routes, "compose_answer", lambda q, ev: _composed("A [chunk:2].", [2]))
    monkeypatch.setattr(
        routes,
        "resolve_citations",
        lambda answer, **kw: ResolvedCitations(citations=[_citation(2)], unresolved_chunk_ids=[]),
    )

    def broken_record(engine: Any, **kwargs: Any) -> None:
        raise OperationalError("INSERT ...", {}, Exception("db gone"))

    monkeypatch.setattr(routes, "record_query", broken_record)
    response = harness.client.post("/ask", json={"question": "q"})
    assert response.status_code == 500
    assert response.json()["detail"] == "query accounting failed"


def test_ask_unknown_debug_value_is_422(harness: _Harness) -> None:
    response = harness.client.post("/ask?debug=everything", json={"question": "anything"})
    assert response.status_code == 422


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_ask_blank_question_is_422_not_a_paid_embedding_call(harness: _Harness, question: str) -> None:
    response = harness.client.post("/ask?debug=retrieval", json={"question": question})
    assert response.status_code == 422
    # Neither retriever ran (the dense leg would have spent on an embedding).
    assert harness.calls == {}


def test_healthz_ok_when_database_answers(harness: _Harness) -> None:
    class _FakeConn:
        def execute(self, *args: Any, **kwargs: Any) -> None:
            return None

        def __enter__(self) -> _FakeConn:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    class _FakeEngine:
        def connect(self) -> _FakeConn:
            return _FakeConn()

    harness.app.dependency_overrides[get_engine] = lambda: _FakeEngine()
    response = harness.client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_healthz_maps_database_failure_to_503(harness: _Harness) -> None:
    class _BrokenEngine:
        def connect(self) -> Any:
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    harness.app.dependency_overrides[get_engine] = lambda: _BrokenEngine()
    response = harness.client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["detail"] == "database unreachable"


def _configure_startup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete request-path config: DB URL nobody listens on + dummy provider.

    create_engine pools lazily and get_client makes no network call, so the
    lifespan opens no connection and spends nothing — these stay DB-free.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x:x@127.0.0.1:9/analyst")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("ANALYST_MODEL", "gpt-5.4-mini")  # priced; plain /ask needs it


def test_lifespan_creates_and_disposes_the_shared_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_startup_env(monkeypatch)
    disposed: list[Engine] = []
    original_dispose = Engine.dispose

    def spy(self: Engine, close: bool = True) -> None:
        disposed.append(self)
        original_dispose(self, close)

    monkeypatch.setattr(Engine, "dispose", spy)
    app = create_app()
    with TestClient(app):
        engine = app.state.engine
        assert isinstance(engine, Engine)
    assert disposed == [engine]


def test_lifespan_fails_loud_without_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"), TestClient(create_app()):
        pass  # pragma: no cover — startup must raise before we get here


def test_lifespan_fails_loud_without_provider_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # A keyless API would pass /healthz but 500 on every /ask?debug=retrieval
    # (the dense leg embeds per request) — so startup must refuse instead.
    _configure_startup_env(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"), TestClient(create_app()):
        pass  # pragma: no cover — startup must raise before we get here


def test_lifespan_fails_loud_without_analyst_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Plain /ask ALWAYS generates now, so a retrieval-only env (no ANALYST_MODEL)
    # that get_client tolerates for ingest must still be refused at startup —
    # otherwise the app reports healthy then 500s on the first answerable /ask.
    _configure_startup_env(monkeypatch)
    monkeypatch.delenv("ANALYST_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="ANALYST_MODEL"), TestClient(create_app()):
        pass  # pragma: no cover — startup must raise before we get here


def test_lifespan_loads_the_source_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    # H1-20 loads the citation registry ONCE at startup (get_registry then injects
    # app.state.registry). Lock the wiring: a refactor that moved the load into a
    # per-request get_registry (reintroducing a first-/ask disk read) must fail here.
    from wealthlens_analyst.answer.citations import SourceMeta

    _configure_startup_env(monkeypatch)
    app = create_app()
    with TestClient(app):
        registry = app.state.registry
    assert registry and all(isinstance(source, SourceMeta) for source in registry.values())


def test_lifespan_fails_loud_on_malformed_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    # A malformed/duplicated corpus source must KILL startup (fail-loud), not
    # 500 the first /ask — invariant 5. load_source_registry itself fail-louds
    # (test_citations.py); here we lock that the lifespan surfaces it at startup.
    import wealthlens_analyst.api.app as app_module

    _configure_startup_env(monkeypatch)

    def broken_registry(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("duplicate analyst_corpus source id 'ons-was-wealth'")

    monkeypatch.setattr(app_module, "load_source_registry", broken_registry)
    with pytest.raises(ValueError, match="duplicate analyst_corpus"), TestClient(create_app()):
        pass  # pragma: no cover — startup must raise before we get here


def test_retrieval_limits_are_the_h1_14_measurement_basis() -> None:
    # H1-14's recall report is defined against these depths; changing either
    # must be a deliberate, review-visible decision, not a drive-by edit.
    assert routes._PER_RETRIEVER_LIMIT == 50
    assert routes._FUSED_LIMIT == 20


def test_ask_debug_retrieval_truncates_to_the_fused_limit(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # 30 disjoint hits per leg -> 60 fused candidates available; the response
    # must carry exactly _FUSED_LIMIT, so the top-N contract is behavioural,
    # not just a constant.
    lexical = [_hit(i, rank=i, score=1.0 / i) for i in range(1, 31)]
    dense = [_hit(100 + i, rank=i, score=1.0 / i) for i in range(1, 31)]
    monkeypatch.setattr(routes, "search_fts", lambda q, *, limit, engine: lexical)
    monkeypatch.setattr(routes, "search_dense_by_vector", lambda v, *, limit, engine: dense)
    response = harness.client.post("/ask?debug=retrieval", json={"question": "deep corpus"})
    assert response.status_code == 200
    assert len(response.json()["candidates"]) == routes._FUSED_LIMIT


def test_ask_question_with_nul_byte_is_422_not_a_db_error(harness: _Harness) -> None:
    response = harness.client.post("/ask?debug=retrieval", json={"question": "wealth\x00share"})
    assert response.status_code == 422
    assert harness.calls == {}


def test_ask_connection_failure_is_503_and_skips_the_doomed_error_row(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OperationalError = connection-level: the accounting write would hit the
    # same dead database, so the route must not stack a second connect-timeout
    # onto the caller's wait for the 503.
    def broken_fts(query: str, *, limit: int, engine: Any) -> list[ChunkHit]:
        raise OperationalError("SELECT ...", {}, Exception("connection refused"))

    monkeypatch.setattr(routes, "search_fts", broken_fts)
    response = harness.client.post("/ask?debug=retrieval", json={"question": "anything"})
    assert response.status_code == 503
    assert response.json()["detail"] == "retrieval backend unavailable"
    # The free leg failed first, so the paid embedding call never happened...
    assert "embed" not in harness.calls
    # ...and the doomed write was skipped (nothing was spent to lose).
    assert harness.recorded == []


def test_ask_statement_failure_after_embed_records_error_row_with_real_spend(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dense leg dies AFTER the paid embed, with the connection still
    # alive (statement-level DatabaseError, not OperationalError): the error
    # row must carry the REAL spend — the invariant the `embedded` threading
    # in the route exists for.
    def broken_dense(query_vec: list[float], *, limit: int, engine: Any) -> list[ChunkHit]:
        raise DatabaseError("SELECT ...", {}, Exception("statement failed"))

    monkeypatch.setattr(routes, "search_dense_by_vector", broken_dense)
    response = harness.client.post("/ask?debug=retrieval", json={"question": "anything"})
    assert response.status_code == 503
    assert len(harness.recorded) == 1
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.ERROR
    assert row["tokens_in"] == _EMBED_TOKENS
    assert row["cost_gbp"] == _EMBED_COST_GBP


def test_ask_provider_failure_records_a_zero_spend_error_row_before_the_500(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The embed itself explodes (provider outage): nothing was spent, an
    # `error` row is still recorded, and the exception propagates (500 —
    # provider error schemas land in H1-20).
    class _BrokenClient:
        def embed(self, texts: list[str]) -> EmbeddingResult:
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(routes, "get_client", lambda: _BrokenClient())
    client = TestClient(harness.app, raise_server_exceptions=False)
    response = client.post("/ask?debug=retrieval", json={"question": "anything"})
    assert response.status_code == 500
    assert len(harness.recorded) == 1
    row = harness.recorded[0]
    assert row["decision"] == QueryDecision.ERROR
    assert row["tokens_in"] == 0
    assert row["cost_gbp"] == 0.0


def test_ask_error_row_write_failure_does_not_mask_the_503(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # Double failure with the connection still alive: the dense statement
    # fails AND the accounting insert fails. The caller must still see the
    # 503, not the accounting exception.
    def broken_dense(query_vec: list[float], *, limit: int, engine: Any) -> list[ChunkHit]:
        raise DatabaseError("SELECT ...", {}, Exception("statement failed"))

    def broken_record(engine: Any, **kwargs: Any) -> None:
        raise OperationalError("INSERT ...", {}, Exception("write failed"))

    monkeypatch.setattr(routes, "search_dense_by_vector", broken_dense)
    monkeypatch.setattr(routes, "record_query", broken_record)
    response = harness.client.post("/ask?debug=retrieval", json={"question": "anything"})
    assert response.status_code == 503
    assert response.json()["detail"] == "retrieval backend unavailable"
