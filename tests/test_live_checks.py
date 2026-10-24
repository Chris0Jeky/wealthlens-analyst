"""Exercise guarded live eval entry points with SQLite and fake providers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

import wealthlens_analyst.answer.compose as compose
import wealthlens_analyst.api.app as api_app
import wealthlens_analyst.api.routes as routes
import wealthlens_analyst.budget.middleware as middleware
from wealthlens_analyst.answer.citations import SourceMeta
from wealthlens_analyst.budget.bounds import MAX_PROMPT_BYTES
from wealthlens_analyst.budget.models import QueryDecision, record_query
from wealthlens_analyst.db import chunks_table, query_log_table
from wealthlens_analyst.llm.client import CompletionResult, EmbeddingResult
from wealthlens_analyst.retrieval.fts import ChunkHit

_NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)


@dataclass
class _FakeClient:
    text: str = "A [chunk:9118]."
    calls: list[str] = field(default_factory=list)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append("embed")
        return EmbeddingResult([[0.1]], "fake-embedding", 7, 0.00000025)

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
        self.calls.append("complete")
        return CompletionResult(self.text, "fake-model", 100, 20, 0.001, finish_reason="stop")


@dataclass
class _Harness:
    check: ModuleType
    engine: Engine
    provider: _FakeClient
    evidence: list[ChunkHit]


@pytest.fixture(params=["check_compose_live", "check_citations_live"])
def harness(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Harness]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "evals" / "checks"))
    check = importlib.import_module(request.param)
    monkeypatch.setattr(sys, "argv", [request.param])
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("ANALYST_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "10")
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    metadata = sa.MetaData()
    log = query_log_table.to_metadata(metadata)
    log.append_column(sa.Column("asked_at", sa.DateTime, server_default=sa.text("'2026-10-15 12:00:00'")))
    chunks = chunks_table.to_metadata(metadata)
    chunks.append_column(sa.Column("chunk_id", sa.Integer, primary_key=True))
    budgets = sa.Table(
        "budgets",
        metadata,
        sa.Column("period_start", sa.Date),
        sa.Column("period_end", sa.Date),
        sa.Column("cap_gbp", sa.Numeric(8, 2)),
        sa.Column("active", sa.Boolean),
    )
    metadata.create_all(engine)
    evidence = [
        ChunkHit(i, "test-source", "test-document", "section", None, None, "evidence", rank, 1.0)
        for rank, i in enumerate([9118, 9119], start=1)
    ]
    with engine.begin() as conn:
        conn.execute(
            budgets.insert(),
            [
                {
                    "period_start": date(2026, 10, 1),
                    "period_end": date(2026, 11, 1),
                    "cap_gbp": Decimal("10"),
                    "active": True,
                }
            ],
        )
        conn.execute(
            chunks.insert(),
            [
                {
                    "chunk_id": hit.chunk_id,
                    "source_id": hit.source_id,
                    "document_id": hit.document_id,
                    "section": hit.section,
                    "text": hit.text,
                    "token_count": 1,
                    "access_date": date(2026, 1, 1),
                }
                for hit in evidence
            ],
        )
    provider = _FakeClient()
    monkeypatch.setattr(api_app, "engine_from_settings", lambda settings: engine)
    monkeypatch.setattr(api_app, "get_client", lambda: provider)
    monkeypatch.setattr(
        api_app,
        "load_source_registry",
        lambda: {"test-source": SourceMeta("Test source", "https://example.test/source")},
    )
    monkeypatch.setattr(routes, "get_client", lambda: provider)
    monkeypatch.setattr(compose, "get_client", lambda: provider)
    monkeypatch.setattr(routes, "search_fts", lambda *args, **kwargs: evidence)
    monkeypatch.setattr(routes, "search_dense_by_vector", lambda *args, **kwargs: evidence)
    real_guard = middleware.budget_guard
    monkeypatch.setattr(routes, "budget_guard", lambda engine, question: real_guard(engine, question, now=_NOW))
    # Keep the in-memory database for assertions after the real app lifespan exits.
    dispose = engine.dispose
    monkeypatch.setattr(engine, "dispose", lambda: None)
    try:
        yield _Harness(check, engine, provider, evidence)
    finally:
        dispose()


@pytest.mark.parametrize(
    ("text", "expected_exit", "decision"),
    [
        ("A [chunk:9118].", 0, "answered"),
        ("A [chunk:9118]. Unsupported [chunk:99999999999999999999].", 1, "refused"),
        ("A [chunk:9118]. Drift [chunk: 9118].", 1, "refused"),
        ("A [chunk:9118]. Fabricated [chunk:9999].", 1, "refused"),
        ("No citations.", 1, "refused"),
        ("", 1, "error"),
    ],
    ids=["valid", "invalid-marker", "drift-marker", "fabricated", "uncited", "empty-generation"],
)
def test_live_check_accounts_paid_answer_even_when_it_fails(
    harness: _Harness,
    capsys: pytest.CaptureFixture[str],
    text: str,
    expected_exit: int,
    decision: str,
) -> None:
    harness.provider.text = text
    assert harness.check.main() == expected_exit
    assert harness.provider.calls == ["embed", "complete"]
    with harness.engine.connect() as conn:
        row = conn.execute(sa.select(query_log_table)).mappings().one()
    assert row["decision"] == decision
    assert row["tokens_in"] == 107
    assert row["tokens_out"] == 20
    assert row["cost_gbp"] == Decimal("0.00100025")
    assert middleware._in_flight_reserved == 0
    output = capsys.readouterr().out
    assert ("OK:" in output) == (expected_exit == 0)
    assert ("FAIL:" in output) == (expected_exit != 0)
    if expected_exit == 0 and harness.check.__name__ == "check_citations_live":
        assert "stripped and reported unresolved (caught)" in output


@pytest.mark.parametrize("budget_state", ["exhausted", "insufficient-headroom", "missing-cap", "missing-row"])
def test_live_check_refuses_before_any_provider_call(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    budget_state: str,
) -> None:
    if budget_state in {"exhausted", "insufficient-headroom"}:
        record_query(
            harness.engine,
            question="seed",
            decision=QueryDecision.ANSWERED,
            tokens_in=1,
            tokens_out=0,
            cost_gbp=10 if budget_state == "exhausted" else 9.999,
            latency_ms=1,
        )
    elif budget_state == "missing-cap":
        monkeypatch.delenv("BUDGET_MONTHLY_CAP_GBP")
    else:
        with harness.engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM budgets"))

    def no_retrieval(*args: object, **kwargs: object) -> None:
        pytest.fail("Budget refusal must precede retrieval")

    monkeypatch.setattr(routes, "search_fts", no_retrieval)
    assert harness.check.main() == 1
    assert harness.provider.calls == []
    output = capsys.readouterr().out
    assert "FAIL: over_budget" in output
    assert "before any provider call" in output
    assert "OK:" not in output
    with harness.engine.connect() as conn:
        row = (
            conn.execute(sa.select(query_log_table).where(query_log_table.c.decision == "over_budget")).mappings().one()
        )
    assert row["cost_gbp"] == row["tokens_in"] == row["tokens_out"] == 0
    assert middleware._in_flight_reserved == 0


@pytest.mark.parametrize("evidence_state", ["empty", "oversized"])
def test_live_check_accounts_embedding_only_refusal(
    harness: _Harness,
    capsys: pytest.CaptureFixture[str],
    evidence_state: str,
) -> None:
    if evidence_state == "empty":
        harness.evidence.clear()
    else:
        from dataclasses import replace

        harness.evidence[0] = replace(harness.evidence[0], text="x" * MAX_PROMPT_BYTES)
    assert harness.check.main() == 1
    assert harness.provider.calls == ["embed"]
    with harness.engine.connect() as conn:
        row = conn.execute(sa.select(query_log_table)).mappings().one()
    assert row["decision"] == "refused"
    assert row["tokens_in"] == 7
    assert row["tokens_out"] == 0
    assert row["cost_gbp"] == Decimal("0.00000025")
    assert "FAIL:" in capsys.readouterr().out


def test_live_check_fails_when_accounting_write_fails(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken_write(*args: object, **kwargs: object) -> None:
        raise SQLAlchemyError("fixture accounting failure")

    monkeypatch.setattr(routes, "record_query", broken_write)
    assert harness.check.main() == 1
    assert harness.provider.calls == ["embed", "complete"]
    assert "FAIL: guarded /ask returned HTTP 500" in capsys.readouterr().out
    assert middleware._in_flight_reserved == 0


def test_live_check_fails_if_ask_ever_serves_an_uncited_answer(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defence in depth: the checks must not rely solely on /ask refusing uncited answers."""
    harness.provider.text = "A [chunk:9118]."
    real_validate = harness.check.AnswerResponse.model_validate

    def strip_citations(payload: object) -> object:
        return real_validate(payload).model_copy(update={"citations": []})

    monkeypatch.setattr(harness.check.AnswerResponse, "model_validate", strip_citations)
    assert harness.check.main() == 1
    assert "FAIL: the served answer carries no citations" in capsys.readouterr().out
