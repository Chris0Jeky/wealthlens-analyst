"""H1-29: hand-checked metrics, SQLite fallback and CI Postgres aggregates."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.pool import StaticPool

import wealthlens_analyst.api.routes as routes
import wealthlens_analyst.budget.metrics as metrics
from wealthlens_analyst.api.app import create_app
from wealthlens_analyst.budget.metrics import QueryMetric, read_monthly_metrics, summarize_queries
from wealthlens_analyst.budget.models import QueryDecision

_START = datetime(2026, 12, 1, tzinfo=UTC)
_END = datetime(2027, 1, 1, tzinfo=UTC)
_NOW = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)

# No question/hash columns: the public aggregate must never even read them.
_LOG = sa.Table(
    "query_log",
    sa.MetaData(),
    sa.Column("asked_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("decision", sa.Text, nullable=False),
    sa.Column("latency_ms", sa.Integer, nullable=False),
    sa.Column("cost_gbp", sa.Numeric(12, 8), nullable=False),
)


@pytest.fixture()
def sample_rows() -> list[QueryMetric]:
    # Deliberately unsorted. Served latencies sort to 10,20,...,200, so
    # ceil(.5*20)=10 -> 100 and ceil(.95*20)=19 -> 190 (NOT the maximum).
    served = [
        QueryMetric(
            QueryDecision.ANSWERED if i % 2 else QueryDecision.REFUSED,
            i * 10,
            Decimal(i) / 100,
        )
        for i in range(20, 0, -1)
    ]
    return [
        QueryMetric(QueryDecision.ERROR, 99999, Decimal("0.60")),
        QueryMetric(QueryDecision.OVER_BUDGET, 0, Decimal(0)),
        *served,
        QueryMetric(QueryDecision.ERROR, 1, Decimal(0)),
    ]


def test_seeded_fixture_math_has_hand_checked_percentiles_and_costs(sample_rows: list[QueryMetric]) -> None:
    summary = summarize_queries(iter(sample_rows))
    assert summary.total_queries == 23
    assert summary.decision_counts == {"answered": 10, "refused": 10, "over_budget": 1, "error": 2}
    assert summary.latency_sample_count == 20
    assert summary.latency_p50_ms == 100
    assert summary.latency_p95_ms == 190
    # ALL costs sum to 2.10 + .60 = 2.70; mean = 2.70/23.
    # Sorted costs: 0,0,.01,...,.20,.60. The 12th of 23 is .10.
    assert summary.cost_mean_gbp == Decimal("0.11739130")
    assert summary.cost_median_gbp == Decimal("0.10000000")


def test_empty_math_returns_nulls_and_all_zero_counts() -> None:
    summary = summarize_queries([])
    assert summary.total_queries == summary.latency_sample_count == 0
    assert summary.decision_counts == {"answered": 0, "refused": 0, "over_budget": 0, "error": 0}
    assert summary.latency_p50_ms is summary.latency_p95_ms is None
    assert summary.cost_mean_gbp is summary.cost_median_gbp is None


def test_single_row_preserves_sub_micro_pound_cost() -> None:
    summary = summarize_queries([QueryMetric(QueryDecision.REFUSED, 7, Decimal("0.00000003"))])
    assert summary.total_queries == summary.latency_sample_count == 1
    assert summary.latency_p50_ms == summary.latency_p95_ms == 7
    assert summary.cost_mean_gbp == summary.cost_median_gbp == Decimal("0.00000003")


@pytest.mark.parametrize("costs", [("0.00000001", "0.00000002"), ("0.00000002", "0.00000003")])
def test_even_cost_median_and_mean_round_half_even(costs: tuple[str, str]) -> None:
    summary = summarize_queries([QueryMetric(QueryDecision.ANSWERED, 10, Decimal(cost)) for cost in costs])
    assert summary.cost_mean_gbp == summary.cost_median_gbp == Decimal("0.00000002")


def test_errors_and_cap_refusals_have_costs_but_no_latency_samples() -> None:
    summary = summarize_queries(
        [QueryMetric(QueryDecision.ERROR, 1000, Decimal("0.60")), QueryMetric(QueryDecision.OVER_BUDGET, 0, Decimal(0))]
    )
    assert summary.latency_sample_count == 0
    assert summary.latency_p50_ms is summary.latency_p95_ms is None
    assert summary.total_queries == 2
    assert summary.cost_mean_gbp == summary.cost_median_gbp == Decimal("0.30000000")


@pytest.fixture(params=["sqlite", "postgresql"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    if request.param == "postgresql":
        url = os.environ.get("WEALTHLENS_TEST_DATABASE_URL")
        if not url:
            pytest.skip("WEALTHLENS_TEST_DATABASE_URL not configured; offline SQL coverage runs on SQLite")
        engine = sa.create_engine(url, poolclass=StaticPool)
        try:
            with engine.connect() as conn, conn.begin() as transaction:
                # Use CI's migrated table and keep every fixture write/DDL
                # inside an outer transaction rolled back after each test.
                conn.execute(sa.text("DELETE FROM query_log"))
                conn.execute(sa.text("SET LOCAL TIME ZONE 'Pacific/Auckland'"))

                @contextmanager
                def connect() -> Iterator[Connection]:
                    yield conn

                @contextmanager
                def begin() -> Iterator[Connection]:
                    with conn.begin_nested():
                        yield conn

                monkeypatch.setattr(engine, "connect", connect)
                monkeypatch.setattr(engine, "begin", begin)
                try:
                    yield engine
                finally:
                    transaction.rollback()
        finally:
            engine.dispose()
        return
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _LOG.create(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def _seed(engine: Engine, rows: list[QueryMetric], *, when: datetime = _NOW) -> None:
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            # The migrated table also requires a question hash, never read by metrics.
            conn.execute(
                sa.text("""
                    INSERT INTO query_log (asked_at, question_sha, decision, latency_ms, cost_gbp)
                    VALUES (:asked_at, 'fixture', :decision, :latency_ms, :cost_gbp)
                """),
                [{"asked_at": when.astimezone(UTC), **asdict(row)} for row in rows],
            )
        else:
            conn.execute(_LOG.insert(), [{"asked_at": when.astimezone(UTC), **asdict(row)} for row in rows])


@pytest.fixture()
def client(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[routes.get_engine] = lambda: engine

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Metrics must not invoke a budget guard, provider, or accounting write")

    monkeypatch.setattr(routes, "budget_guard", forbidden)
    monkeypatch.setattr(routes, "get_client", forbidden)
    monkeypatch.setattr(routes, "record_query", forbidden)
    # Even invalid cap configuration and no budgets table cannot block a free route.
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "0")
    # No context-manager entry for TestClient: follow the existing offline app
    # pattern so startup never reads real DB/provider configuration.
    with patch.object(metrics, "datetime", wraps=datetime) as clock:
        clock.now.return_value = _NOW
        yield TestClient(app)


def test_endpoint_serves_hand_checked_percentiles_over_seeded_rows(
    engine: Engine, client: TestClient, sample_rows: list[QueryMetric]
) -> None:
    _seed(engine, sample_rows)
    response = client.get("/metrics/data")
    assert response.status_code == 200
    assert response.json() == {
        "window_start": "2026-12-01T00:00:00Z",
        "window_end": "2027-01-01T00:00:00Z",
        "total_queries": 23,
        "decision_counts": {"answered": 10, "refused": 10, "over_budget": 1, "error": 2},
        "latency_sample_count": 20,
        "latency_p50_ms": 100,
        "latency_p95_ms": 190,
        "cost_mean_gbp": "0.11739130",
        "cost_median_gbp": "0.10000000",
    }
    # Metrics reads must not produce query_log rows (or modify existing rows).
    with engine.connect() as conn:
        assert conn.execute(sa.select(sa.func.count()).select_from(_LOG)).scalar_one() == 23
        assert conn.execute(sa.select(sa.func.sum(_LOG.c.cost_gbp))).scalar_one() == Decimal("2.70000000")


def test_endpoint_empty_window(engine: Engine, client: TestClient) -> None:
    _seed(engine, [QueryMetric(QueryDecision.ANSWERED, 99, Decimal(1))], when=_START - timedelta(microseconds=1))
    response = client.get("/metrics/data")
    assert response.status_code == 200
    body = response.json()
    assert body["total_queries"] == body["latency_sample_count"] == 0
    assert body["decision_counts"] == {"answered": 0, "refused": 0, "over_budget": 0, "error": 0}
    for key in ("latency_p50_ms", "latency_p95_ms", "cost_mean_gbp", "cost_median_gbp"):
        assert body[key] is None


def test_half_open_month_boundaries_and_utc_year_rollover(engine: Engine) -> None:
    for when, latency in [
        (_START - timedelta(microseconds=1), 999),
        (_START, 10),
        (_END - timedelta(microseconds=1), 30),
        (_END, 999),
    ]:
        _seed(engine, [QueryMetric(QueryDecision.ANSWERED, latency, Decimal(latency))], when=when)
    # Local January is still December UTC. Last-microsecond row is inside the
    # month even though it is later than now, matching the budget spend query.
    now = datetime(2027, 1, 1, 0, 30, tzinfo=timezone(timedelta(hours=1)))
    start, end, summary = read_monthly_metrics(engine, now=now)
    assert (start, end) == (_START, _END)
    assert summary.total_queries == 2
    assert summary.latency_p50_ms == 10  # nearest rank, NOT interpolated 20
    assert summary.latency_p95_ms == 30
    assert summary.cost_mean_gbp == summary.cost_median_gbp == Decimal("20.00000000")


# The Postgres fixture already owns a connection; exercise real connect events offline.
@pytest.mark.parametrize("engine", ["sqlite"], indirect=True)
@pytest.mark.parametrize("failure_at", ["connect", "execute"])
def test_database_failure_returns_503_without_leaking_exception(
    engine: Engine, client: TestClient, caplog: pytest.LogCaptureFixture, failure_at: str
) -> None:
    def fail(*args: Any) -> None:
        error = OperationalError if failure_at == "connect" else DatabaseError
        raise error("SELECT", {}, Exception("sensitive-connection-detail"))

    sa.event.listen(engine, "engine_connect" if failure_at == "connect" else "before_cursor_execute", fail)
    response = client.get("/metrics/data")
    assert response.status_code == 503
    assert response.json() == {"detail": "database unreachable"}
    assert "sensitive-connection-detail" not in caplog.text + response.text


def test_openapi_documents_metrics_response() -> None:
    schema = create_app().openapi()
    response = schema["paths"]["/metrics/data"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response == {"$ref": "#/components/schemas/MetricsResponse"}
    assert (
        "Nearest-rank"
        in schema["components"]["schemas"]["MetricsResponse"]["properties"]["latency_p95_ms"]["description"]
    )


@pytest.mark.parametrize("size", [1, 2, 3, 19, 20, 21, 40, 41])
def test_sql_percentile_disc_matches_nearest_rank_for_odd_even_and_tied_samples(engine: Engine, size: int) -> None:
    rows = [QueryMetric(QueryDecision.ANSWERED, i // 2, Decimal(i) / 100) for i in range(size, 0, -1)]
    rows += [QueryMetric(QueryDecision.ERROR, 9999, Decimal("0.25"))]
    _seed(engine, rows)
    assert read_monthly_metrics(engine, now=_NOW)[2] == summarize_queries(rows)


@pytest.mark.parametrize(
    "costs",
    [
        ("0.00000001", "0.00000002"),
        ("0.00000002", "0.00000003"),
        ("9999.99999998", "9999.99999999"),
        ("0.00000001", "0.00000002", "0.00000003"),
    ],
)
def test_sql_cost_mean_and_median_are_exact_half_even(engine: Engine, costs: tuple[str, ...]) -> None:
    rows = [QueryMetric(QueryDecision.OVER_BUDGET, 0, Decimal(cost)) for cost in costs]
    _seed(engine, rows)
    assert read_monthly_metrics(engine, now=_NOW)[2] == summarize_queries(rows)


def test_cache_hit_avoids_all_database_queries(engine: Engine, client: TestClient) -> None:
    _seed(engine, [QueryMetric(QueryDecision.ANSWERED, 7, Decimal("0.25"))])
    with patch.object(engine, "begin", wraps=engine.begin) as begin:
        first = client.get("/metrics/data")
        assert first.status_code == 200
        assert begin.call_count == 1
        with patch.object(engine, "begin", side_effect=AssertionError("Cache hit accessed the database")):
            second = client.get("/metrics/data")
    assert second.json() == first.json()


def test_cache_expires_after_30_seconds(engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(metrics, "monotonic", lambda: clock[0])
    assert client.get("/metrics/data").json()["total_queries"] == 0
    _seed(engine, [QueryMetric(QueryDecision.REFUSED, 12, Decimal("0.10"))])
    clock[0] = 129.999
    assert client.get("/metrics/data").json()["total_queries"] == 0
    clock[0] = 130.0
    assert client.get("/metrics/data").json()["total_queries"] == 1


def test_cache_key_includes_window_start(engine: Engine) -> None:
    _seed(engine, [QueryMetric(QueryDecision.ANSWERED, 7, Decimal("0.25"))], when=_END)
    assert read_monthly_metrics(engine, now=_NOW)[2].total_queries == 0
    start, _, summary = read_monthly_metrics(engine, now=_END)
    assert start == _END
    assert summary.total_queries == 1


def test_expired_cache_failure_is_503_and_is_not_cached(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = [100.0]
    monkeypatch.setattr(metrics, "monotonic", lambda: clock[0])
    assert client.get("/metrics/data").status_code == 200
    clock[0] = 130.0
    with patch.object(engine, "begin", side_effect=OperationalError("SELECT", {}, Exception("sensitive-detail"))):
        for _ in range(2):
            response = client.get("/metrics/data")
            assert response.status_code == 503
            assert response.json() == {"detail": "database unreachable"}
    assert "sensitive-detail" not in caplog.text
    assert client.get("/metrics/data").status_code == 200


@pytest.mark.parametrize("engine", ["sqlite"], indirect=True)
def test_sqlite_fallback_fetches_only_ordered_latency_and_middle_costs(engine: Engine) -> None:
    _seed(engine, [QueryMetric(QueryDecision.ANSWERED, i, Decimal(i)) for i in range(10)])
    statements: list[str] = []

    def capture(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", capture)
    read_monthly_metrics(engine, now=_NOW)
    assert len(statements) == 3
    assert "count(*)" in statements[0] and "sum(" in statements[0]
    assert statements[1].startswith("SELECT query_log.cost_gbp ")
    assert "ORDER BY query_log.cost_gbp" in statements[1] and "LIMIT" in statements[1]
    assert statements[2].startswith("SELECT query_log.latency_ms ")
    assert "ORDER BY query_log.latency_ms" in statements[2]


@pytest.mark.parametrize("engine", ["postgresql"], indirect=True)
def test_postgres_aggregates_in_one_select_with_transaction_local_timeout(engine: Engine) -> None:
    _seed(engine, [QueryMetric(QueryDecision.ANSWERED, 7, Decimal("0.25"))])
    statements: list[str] = []
    timeouts: list[str] = []

    def capture(conn: Connection, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)
        if "percentile_disc" in statement:
            timeouts.append(conn.exec_driver_sql("SHOW statement_timeout").scalar_one())

    sa.event.listen(engine, "before_cursor_execute", capture)
    assert read_monthly_metrics(engine, now=_NOW)[2].total_queries == 1
    selects = [statement for statement in statements if statement.startswith("SELECT")]
    assert len(selects) == 1
    assert selects[0].count("percentile_disc") == 4
    assert "SET LOCAL statement_timeout = '2s'" in statements
    assert timeouts == ["2s"]


@pytest.mark.parametrize("engine", ["postgresql"], indirect=True)
def test_postgres_timeout_returns_503_and_restores_setting(engine: Engine, client: TestClient) -> None:
    with engine.connect() as conn:
        original_timeout = conn.exec_driver_sql("SHOW statement_timeout").scalar_one()
    sqlstates: list[str] = []

    def slow_aggregate(conn: Any, cursor: Any, statement: str, parameters: Any, *args: Any) -> tuple[str, Any]:
        if "percentile_disc" in statement:
            return "SELECT pg_sleep(3)", {}
        return statement, parameters

    def capture_error(context: Any) -> None:
        sqlstates.append(context.original_exception.sqlstate)

    sa.event.listen(engine, "before_cursor_execute", slow_aggregate, retval=True)
    sa.event.listen(engine, "handle_error", capture_error)
    try:
        response = client.get("/metrics/data")
    finally:
        sa.event.remove(engine, "before_cursor_execute", slow_aggregate)
        sa.event.remove(engine, "handle_error", capture_error)
    assert sqlstates == ["57014"]  # PostgreSQL query_canceled, not a SQL/connection error.
    assert response.status_code == 503
    assert response.json() == {"detail": "database unreachable"}
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SHOW statement_timeout").scalar_one() == original_timeout
        assert conn.exec_driver_sql("SELECT 1").scalar_one() == 1


@pytest.mark.parametrize("engine", ["sqlite"], indirect=True)
def test_cache_does_not_reuse_another_engines_result(engine: Engine) -> None:
    _seed(engine, [QueryMetric(QueryDecision.ANSWERED, 7, Decimal("0.25"))])
    other = sa.create_engine("sqlite://")
    try:
        _LOG.create(other)
        assert read_monthly_metrics(engine, now=_NOW)[2].total_queries == 1
        assert read_monthly_metrics(other, now=_NOW)[2].total_queries == 0
    finally:
        other.dispose()
