"""H1-27: seeded accounting -> exact 429, with no Postgres or model calls.

The budget queries use portable Core SQL, so SQLite exercises real filtering,
aggregation and inserts offline. The same tests optionally run on Postgres via
WEALTHLENS_TEST_DATABASE_URL, using session-local temporary tables only.
Retrieval/provider seams alone are faked.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from threading import Event
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

import wealthlens_analyst.api.routes as routes
import wealthlens_analyst.budget.middleware as middleware
from wealthlens_analyst.answer.abstain import GateDecision
from wealthlens_analyst.answer.compose import EmptyGenerationError
from wealthlens_analyst.api.app import create_app
from wealthlens_analyst.api.schemas import OverBudgetResponse
from wealthlens_analyst.budget.bounds import request_cost_bound_gbp
from wealthlens_analyst.budget.models import (
    BudgetUnavailableError,
    QueryDecision,
    active_budget_cap_gbp,
    current_spend_gbp,
    record_query,
)
from wealthlens_analyst.config import load_settings
from wealthlens_analyst.llm.client import CompletionResult, EmbeddingResult
from wealthlens_analyst.retrieval.fts import ChunkHit

_NOW = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
_QUESTION = "How is wealth distributed?"
_BODY = {"mode": "over_budget", "question": _QUESTION, "reason": "over_budget"}


@pytest.fixture(params=["sqlite", "postgresql"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("ANALYST_MODEL", "gpt-5.4-mini")
    if request.param == "postgresql":
        url = os.environ.get("WEALTHLENS_TEST_DATABASE_URL")
        if not url:
            pytest.skip("WEALTHLENS_TEST_DATABASE_URL not configured; offline SQL coverage runs on SQLite")
        engine = sa.create_engine(url, poolclass=StaticPool)
        timestamp_type = "TIMESTAMP WITH TIME ZONE"
    else:
        engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        timestamp_type = "TIMESTAMP"
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            # Exercise UTC boundaries even when the DB session uses another zone.
            conn.execute(sa.text("SET TIME ZONE 'Pacific/Auckland'"))
        conn.execute(
            sa.text("""
            CREATE TEMPORARY TABLE budgets (
                period_start DATE NOT NULL, period_end DATE NOT NULL,
                cap_gbp NUMERIC(8, 2) NOT NULL, active BOOLEAN NOT NULL
            )
        """)
        )
        # Freeze BOTH clocks: admission and query_log's server default. Real
        # CURRENT_TIMESTAMP would put new spend outside the guard's frozen month.
        stored_now = _NOW.isoformat() if engine.dialect.name == "postgresql" else _NOW.strftime("%Y-%m-%d %H:%M:%S.%f")
        conn.execute(
            sa.text(f"""
            CREATE TEMPORARY TABLE query_log (
                asked_at {timestamp_type} NOT NULL DEFAULT '{stored_now}',
                question_sha TEXT NOT NULL, decision TEXT NOT NULL,
                gate_signal FLOAT, tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL, cost_gbp NUMERIC(12, 8) NOT NULL,
                latency_ms INTEGER NOT NULL, rerank_used BOOLEAN NOT NULL
            )
        """)
        )
        conn.execute(
            sa.text("""
            INSERT INTO budgets VALUES ('2026-12-01', '2027-01-01', 10, true)
        """)
        )
    yield engine
    engine.dispose()


def _seed_spend(
    engine: Engine, amount: float, when: datetime = _NOW, decision: QueryDecision = QueryDecision.ANSWERED
) -> None:
    record_query(engine, question="seed", decision=decision, tokens_in=1, tokens_out=0, cost_gbp=amount, latency_ms=1)
    at = when.astimezone(UTC)
    stored_at = at.isoformat() if engine.dialect.name == "postgresql" else at.strftime("%Y-%m-%d %H:%M:%S.%f")
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE query_log SET asked_at = :at WHERE question_sha = :sha"),
            {"at": stored_at, "sha": hashlib.sha256(b"seed").hexdigest()},
        )


@pytest.fixture()
def client(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "10.00")
    app = create_app()
    app.dependency_overrides[routes.get_engine] = lambda: engine
    app.dependency_overrides[routes.get_registry] = lambda: {}
    real_guard = middleware.budget_guard
    monkeypatch.setattr(routes, "budget_guard", lambda engine, question: real_guard(engine, question, now=_NOW))

    def no_provider() -> Any:
        pytest.fail("A refused request must never reach the model client")

    monkeypatch.setattr(routes, "get_client", no_provider)
    return TestClient(app)


@pytest.mark.parametrize("path", ["/ask", "/ask?debug=retrieval"])
@pytest.mark.parametrize("spend", [10.0, 10.01])
def test_exhausted_seeded_budget_returns_exact_429(
    engine: Engine, client: TestClient, path: str, spend: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_spend(engine, spend)

    def no_retrieval(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Exhausted budget must be rejected before retrieval")

    monkeypatch.setattr(routes, "search_fts", no_retrieval)
    response = client.post(path, json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY
    assert OverBudgetResponse.model_validate(response.json()).model_dump() == _BODY
    with engine.connect() as conn:
        refusal = conn.execute(sa.text("SELECT * FROM query_log WHERE decision = 'over_budget'")).mappings().one()
    assert refusal["tokens_in"] == refusal["tokens_out"] == refusal["cost_gbp"] == 0


@pytest.mark.parametrize("path", ["/ask", "/ask?debug=retrieval"])
def test_under_cap_passes_and_records_embedding_spend(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    _seed_spend(engine, 9.95)
    calls: list[str] = []

    class FakeClient:
        def embed(self, texts: list[str]) -> EmbeddingResult:
            calls.extend(texts)
            return EmbeddingResult(vectors=[[0.1]], model="fake", tokens_in=1, cost_gbp=0.001)

    monkeypatch.setattr(routes, "get_client", FakeClient)
    monkeypatch.setattr(routes, "search_fts", lambda *args, **kwargs: [])
    monkeypatch.setattr(routes, "search_dense_by_vector", lambda *args, **kwargs: [])
    response = client.post(path, json={"question": _QUESTION})
    assert response.status_code == 200
    assert calls == [_QUESTION]
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM query_log")).scalar_one() == 2


@pytest.mark.parametrize("cap", [None, "", "not-a-number", "nan", "inf", "0", "-1"])
def test_missing_or_invalid_config_refuses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, cap: str | None
) -> None:
    if cap is None:
        monkeypatch.delenv("BUDGET_MONTHLY_CAP_GBP", raising=False)
    else:
        monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", cap)
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        "DELETE FROM budgets",
        "UPDATE budgets SET active = false",
        "UPDATE budgets SET cap_gbp = 0",
        "UPDATE budgets SET period_end = '2026-12-31'",
        "UPDATE budgets SET period_start = '2027-01-01', period_end = '2027-02-01'",
    ],
)
def test_unavailable_budget_row_refuses(engine: Engine, client: TestClient, change: str) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text(change))
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY


@pytest.mark.parametrize("reader", ["active_budget_cap_gbp", "current_spend_gbp"])
def test_database_read_failure_refuses_without_retry_or_error_leak(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, reader: str
) -> None:
    def broken_read(*args: Any, **kwargs: Any) -> Any:
        raise OperationalError("SELECT", {}, RuntimeError("private-database-error"))

    def no_accounting(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Do not retry the failed database to account for zero spend")

    monkeypatch.setattr(middleware, reader, broken_read)
    monkeypatch.setattr(middleware, "record_query", no_accounting)
    response = client.post("/ask?debug=retrieval", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY
    assert "private-database-error" not in caplog.text
    assert client.get("/healthz").status_code == 200


def test_connection_failure_refuses(engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_connect() -> Any:
        raise OperationalError("connect", {}, RuntimeError("unavailable"))

    monkeypatch.setattr(engine, "connect", broken_connect)
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY


def test_refusal_write_failure_keeps_the_429(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_spend(engine, 10)

    def broken_write(*args: Any, **kwargs: Any) -> None:
        raise OperationalError("INSERT", {}, RuntimeError("unavailable"))

    monkeypatch.setattr(middleware, "record_query", broken_write)
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY


def test_persisted_and_configured_caps_both_bind(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_spend(engine, 5)
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "5")
    assert client.post("/ask", json={"question": _QUESTION}).json() == _BODY
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "20")
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE budgets SET cap_gbp = 5"))
    assert client.post("/ask", json={"question": _QUESTION}).json() == _BODY


def test_rollover_resets_spend_only_with_a_new_active_budget(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "10")
    _seed_spend(engine, 10)
    january = datetime(2027, 1, 1, tzinfo=UTC)
    with pytest.raises(middleware.BudgetRefusal), middleware.budget_guard(engine, _QUESTION, now=_NOW):
        pytest.fail("Exhausted December budget must refuse")
    with pytest.raises(middleware.BudgetRefusal), middleware.budget_guard(engine, _QUESTION, now=january):
        pytest.fail("Missing January budget must refuse")
    with engine.connect() as conn:
        assert current_spend_gbp(conn, now=_NOW) == Decimal("10")
        assert current_spend_gbp(conn, now=january) == Decimal("0")
        with pytest.raises(BudgetUnavailableError):
            active_budget_cap_gbp(conn, now=january)
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE budgets SET active = false"))
        conn.execute(sa.text("INSERT INTO budgets VALUES ('2027-01-01', '2027-02-01', 10, true)"))
    with engine.connect() as conn:
        assert active_budget_cap_gbp(conn, now=january) == Decimal("10")
        assert current_spend_gbp(conn, now=january) == Decimal("0")
    with middleware.budget_guard(engine, _QUESTION, now=january):
        pass


@pytest.mark.parametrize("decision", list(QueryDecision))
def test_spend_uses_utc_half_open_month_and_all_decisions(engine: Engine, decision: QueryDecision) -> None:
    # New Year's midnight +01:00 is still December in UTC.
    offset_now = datetime(2027, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    for when, expected in [
        (datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC), Decimal("0")),
        (datetime(2026, 12, 1, tzinfo=UTC), Decimal("0.00000025")),
        (datetime(2027, 1, 1, tzinfo=UTC), Decimal("0")),
    ]:
        with engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM query_log"))
        _seed_spend(engine, 0.00000025, when, decision)
        with engine.connect() as conn:
            assert current_spend_gbp(conn, now=offset_now) == expected


def test_free_routes_and_validation_never_run_guard(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_guard(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Free routes or invalid input must never reach the guard")

    monkeypatch.setattr(routes, "budget_guard", no_guard)
    assert client.get("/healthz").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    # Metrics itself remains H1-29's stub; its route must remain guard-exempt.
    with pytest.raises(NotImplementedError, match="H1-29"):
        client.get("/metrics/data")
    assert client.post("/ask", json={"question": " "}).status_code == 422
    assert client.post("/ask?debug=unknown", json={"question": _QUESTION}).status_code == 422


def _stub_retrieval(monkeypatch: pytest.MonkeyPatch, cost: float = 0.0) -> None:
    class FakeClient:
        def embed(self, texts: list[str]) -> EmbeddingResult:
            return EmbeddingResult(vectors=[[0.1]], model="fake", tokens_in=1, cost_gbp=cost)

    monkeypatch.setattr(routes, "get_client", FakeClient)
    monkeypatch.setattr(routes, "search_fts", lambda *args, **kwargs: [])
    monkeypatch.setattr(routes, "search_dense_by_vector", lambda *args, **kwargs: [])


def test_single_request_cannot_overshoot_remaining_allowance(engine: Engine, client: TestClient) -> None:
    bound = request_cost_bound_gbp(_QUESTION, load_settings())
    _seed_spend(engine, float(Decimal("10") - bound + Decimal("0.00000001")))
    # client fixture fails if any refused request reaches the provider.
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY
    assert middleware._in_flight_reserved == 0


def test_interleaved_requests_reserve_before_either_finishes(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_spend(engine, 9.95)
    entered, finish = Event(), Event()
    bound = request_cost_bound_gbp(_QUESTION, load_settings())
    assert bound <= Decimal("0.05") < 2 * bound
    _stub_retrieval(monkeypatch)

    class HeldClient:
        def embed(self, texts: list[str]) -> EmbeddingResult:
            entered.set()
            assert finish.wait(10), "Timed out waiting for concurrent admission"
            return EmbeddingResult(vectors=[[0.1]], model="fake", tokens_in=1, cost_gbp=float(bound))

    monkeypatch.setattr(routes, "get_client", HeldClient)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/ask", json={"question": _QUESTION})
        try:
            assert entered.wait(10), "First request never reached the provider"
            second = pool.submit(client.post, "/ask", json={"question": _QUESTION}).result(timeout=10)
            assert second.status_code == 429
            assert second.json() == _BODY
            assert middleware._in_flight_reserved == bound
        finally:
            finish.set()
        assert first.result(timeout=10).status_code == 200
    assert middleware._in_flight_reserved == 0
    with engine.connect() as conn:
        assert current_spend_gbp(conn, now=_NOW) == Decimal("9.95") + bound
        assert current_spend_gbp(conn, now=_NOW) <= Decimal("10")
        assert conn.execute(sa.text("SELECT count(*) FROM query_log WHERE decision = 'refused'")).scalar_one() == 1


@pytest.mark.parametrize("path", ["/ask", "/ask?debug=retrieval"])
def test_successive_recorded_requests_exhaust_cap(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    bound = request_cost_bound_gbp(_QUESTION, load_settings())
    _seed_spend(engine, float(Decimal("10") - 2 * bound))
    _stub_retrieval(monkeypatch, float(bound))
    for remaining in (bound, Decimal("0")):
        assert client.post(path, json={"question": _QUESTION}).status_code == 200
        assert middleware._in_flight_reserved == 0
        with engine.connect() as conn:
            assert current_spend_gbp(conn, now=_NOW) == Decimal("10") - remaining
    response = client.post(path, json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY


@pytest.mark.parametrize("outcome", ["success", "refusal", "exception", "accounting_failure"])
def test_reservation_released_after_terminal_accounting(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    bound = request_cost_bound_gbp(_QUESTION, load_settings())
    _seed_spend(engine, float(Decimal("10") - bound))
    _stub_retrieval(monkeypatch)
    recorded: list[QueryDecision] = []
    real_record = routes.record_query

    def checked_record(*args: Any, **kwargs: Any) -> None:
        assert middleware._in_flight_reserved == bound
        recorded.append(kwargs["decision"])
        if outcome == "accounting_failure":
            raise OperationalError("INSERT", {}, RuntimeError("unavailable"))
        real_record(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(routes, "record_query", checked_record)
        if outcome == "exception":

            def broken_client() -> Any:
                raise RuntimeError("fake provider failure")

            patch.setattr(routes, "get_client", broken_client)
            with pytest.raises(RuntimeError, match="fake provider failure"):
                client.post("/ask", json={"question": _QUESTION})
        else:
            path = "/ask" if outcome == "refusal" else "/ask?debug=retrieval"
            response = client.post(path, json={"question": _QUESTION})
            assert response.status_code == (500 if outcome == "accounting_failure" else 200)
    expected = {
        "success": QueryDecision.ANSWERED,
        "refusal": QueryDecision.REFUSED,
        "exception": QueryDecision.ERROR,
        "accounting_failure": QueryDecision.ANSWERED,
    }
    assert recorded == [expected[outcome]]
    assert middleware._in_flight_reserved == 0
    # The same tight allowance is usable again: no reservation survived any exit.
    assert client.post("/ask", json={"question": _QUESTION}).status_code == 200
    assert middleware._in_flight_reserved == 0


@pytest.mark.parametrize("setting", ["EMBEDDING_MODEL", "ANALYST_MODEL"])
def test_unpriced_model_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch, setting: str) -> None:
    monkeypatch.setenv(setting, "unpriced-model")
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 429
    assert response.json() == _BODY
    assert middleware._in_flight_reserved == 0


def test_billed_generation_error_is_recorded_before_reservation_release(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_spend(engine, 9.95)
    _stub_retrieval(monkeypatch)
    monkeypatch.setattr(routes, "evaluate_evidence", lambda *args: GateDecision(True, 1.0, None))
    monkeypatch.setattr(
        routes, "fuse_rrf", lambda *args, **kwargs: [ChunkHit(1, "source", "doc", None, None, None, "text", 1, 1.0)]
    )

    def empty_generation(*args: Any) -> Any:
        raise EmptyGenerationError("empty", CompletionResult("", "fake", 1, 1, 0.001, "length"))

    monkeypatch.setattr(routes, "compose_answer", empty_generation)
    real_record = routes.record_query

    def checked_record(*args: Any, **kwargs: Any) -> None:
        assert middleware._in_flight_reserved == request_cost_bound_gbp(_QUESTION, load_settings())
        real_record(*args, **kwargs)

    monkeypatch.setattr(routes, "record_query", checked_record)
    response = client.post("/ask", json={"question": _QUESTION})
    assert response.status_code == 500
    assert middleware._in_flight_reserved == 0
    with engine.connect() as conn:
        assert current_spend_gbp(conn, now=_NOW) == Decimal("9.951")
        assert conn.execute(sa.text("SELECT decision FROM query_log WHERE decision = 'error'")).scalar_one() == "error"
    assert client.post("/ask?debug=retrieval", json={"question": _QUESTION}).status_code == 200
    assert middleware._in_flight_reserved == 0
