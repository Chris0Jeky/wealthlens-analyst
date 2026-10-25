"""Read-only public metrics from query_log (H1-29, ADR 0002).

The window is [first day of this UTC month, first day of next UTC month),
matching budget.models.current_spend_gbp, including any recorded future rows
inside that month. Counts cover every decision. Latency covers answered and
refused rows (including debug retrieval), excluding errors and cap refusals.
Latency percentiles use nearest rank: sorted_values[ceil(p * n) - 1]. Thus
p50 for an even sample is the lower middle value, not an interpolated median.

Cost/query includes ALL logged requests, including billed errors and zero-cost
cap refusals. Its median averages the two middle costs for an even sample.
Costs stay Decimal, rounded to query_log's eight places with ROUND_HALF_EVEN.
No samples means None, rather than a misleading zero. No questions or hashes
are read, and neither model calls nor accounting writes occur here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from math import ceil
from threading import Lock
from time import monotonic
from weakref import WeakKeyDictionary

import sqlalchemy as sa
from sqlalchemy import Connection, Engine
from sqlalchemy.exc import TimeoutError as SATimeoutError

from wealthlens_analyst.budget.models import QueryDecision

# Read-only projection: db.query_log_table deliberately omits asked_at, which
# the server manages. DDL remains owned by migration 0004_query_log.
_query_log = sa.table(
    "query_log",
    sa.column("asked_at", sa.TIMESTAMP(timezone=True)),
    sa.column("decision", sa.Text),
    sa.column("cost_gbp", sa.Numeric(12, 8)),
    sa.column("latency_ms", sa.Integer),
)
_COST_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True)
class QueryMetric:
    """Only the accounting fields needed for the pure aggregation."""

    decision: QueryDecision
    latency_ms: int
    cost_gbp: Decimal


@dataclass(frozen=True)
class QueryMetrics:
    """Aggregates over rows already selected for a window."""

    total_queries: int
    decision_counts: dict[QueryDecision, int]
    latency_sample_count: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    cost_mean_gbp: Decimal | None
    cost_median_gbp: Decimal | None


def summarize_queries(rows: Iterable[QueryMetric]) -> QueryMetrics:
    """Compute nearest-rank latency and exact Decimal mean/median cost. Pure."""
    counts = {decision: 0 for decision in QueryDecision}
    latencies: list[int] = []
    costs: list[Decimal] = []
    for row in rows:
        counts[row.decision] += 1
        costs.append(row.cost_gbp)
        if row.decision in (QueryDecision.ANSWERED, QueryDecision.REFUSED):
            latencies.append(row.latency_ms)
    latencies.sort()
    costs.sort()
    n = len(costs)
    mean = sum(costs, Decimal(0)) / n if n else None
    median = (costs[(n - 1) // 2] + costs[n // 2]) / 2 if n else None
    return QueryMetrics(
        total_queries=n,
        decision_counts=counts,
        latency_sample_count=len(latencies),
        latency_p50_ms=latencies[ceil(0.50 * len(latencies)) - 1] if latencies else None,
        latency_p95_ms=latencies[ceil(0.95 * len(latencies)) - 1] if latencies else None,
        cost_mean_gbp=mean.quantize(_COST_QUANTUM, rounding=ROUND_HALF_EVEN) if mean is not None else None,
        cost_median_gbp=median.quantize(_COST_QUANTUM, rounding=ROUND_HALF_EVEN) if median is not None else None,
    )


@dataclass(frozen=True)
class _CacheEntry:
    start: datetime
    summary: QueryMetrics
    expires_at: float


# One window per live engine, so rollover cannot grow the cache indefinitely.
# Serialise refreshes as well as hits to prevent simultaneous public requests
# from all scanning an expired window. Failed reads are never cached.
_cache: WeakKeyDictionary[Engine, _CacheEntry] = WeakKeyDictionary()
_cache_lock = Lock()
_CACHE_TTL_SECONDS = 30
_LOCK_WAIT_SECONDS = 2.5


def _read_metrics(conn: Connection, start: datetime, end: datetime) -> QueryMetrics:
    window = sa.and_(_query_log.c.asked_at >= start, _query_log.c.asked_at < end)
    served = _query_log.c.decision.in_((QueryDecision.ANSWERED, QueryDecision.REFUSED))
    postgres = conn.dialect.name == "postgresql"
    cost = _query_log.c.cost_gbp
    # SQLite NUMERIC sums use floats. Sum integer eight-place units there so
    # a floating-point accumulation error cannot change a half-even tie.
    cost_sum = sa.func.sum(cost if postgres else sa.cast(sa.func.round(cost * 100_000_000), sa.BigInteger))
    aggregates = [
        sa.func.count().label("total"),
        cost_sum.label("cost_sum"),
        sa.func.count().filter(served).label("latency_count"),
        *(
            sa.func.count().filter(_query_log.c.decision == decision).label(decision.value)
            for decision in QueryDecision
        ),
    ]
    if postgres:
        # percentile_disc chooses the first value whose cumulative fraction
        # is >= p: rank ceil(p*n), exactly our nearest-rank latency definition.
        # For cost, ascending/descending p50 give the two middle values when
        # n is even (the same middle when odd), without float interpolation.
        aggregates.extend(
            [
                sa.func.percentile_disc(0.50).within_group(_query_log.c.latency_ms).filter(served).label("p50"),
                sa.func.percentile_disc(0.95).within_group(_query_log.c.latency_ms).filter(served).label("p95"),
                sa.func.percentile_disc(0.50).within_group(cost).label("cost_lower"),
                sa.func.percentile_disc(0.50).within_group(cost.desc()).label("cost_upper"),
            ]
        )
    row = conn.execute(sa.select(*aggregates).where(window)).mappings().one()
    n = row["total"]
    mean = None
    median = None
    p50 = p95 = None
    if n:
        total_cost = row["cost_sum"] if postgres else Decimal(row["cost_sum"]) * _COST_QUANTUM
        mean = (total_cost / n).quantize(_COST_QUANTUM, rounding=ROUND_HALF_EVEN)
        if postgres:
            middle = (row["cost_lower"] + row["cost_upper"]) / 2
        else:
            costs = (
                conn.execute(
                    sa.select(cost).where(window).order_by(cost).offset((n - 1) // 2).limit(2 if n % 2 == 0 else 1)
                )
                .scalars()
                .all()
            )
            middle = sum(costs, Decimal(0)) / len(costs)
        median = middle.quantize(_COST_QUANTUM, rounding=ROUND_HALF_EVEN)
    if postgres:
        p50, p95 = row["p50"], row["p95"]
    elif row["latency_count"]:
        latencies = (
            conn.execute(sa.select(_query_log.c.latency_ms).where(window, served).order_by(_query_log.c.latency_ms))
            .scalars()
            .all()
        )
        p50 = latencies[ceil(0.50 * len(latencies)) - 1]
        p95 = latencies[ceil(0.95 * len(latencies)) - 1]
    return QueryMetrics(
        total_queries=n,
        decision_counts={decision: row[decision.value] for decision in QueryDecision},
        latency_sample_count=row["latency_count"],
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        cost_mean_gbp=mean,
        cost_median_gbp=median,
    )


def read_monthly_metrics(engine: Engine, *, now: datetime | None = None) -> tuple[datetime, datetime, QueryMetrics]:
    """Read UTC month metrics, cached in process for 30s per engine/window start.

    Cached values may lag writes for up to 30s; month rollover bypasses them.
    PostgreSQL reads have a transaction-local 2s statement timeout, which does
    not persist on the pooled connection. Database failures propagate without
    caching a failure or serving an expired value. SQLite has no SET LOCAL.
    """
    now = now if now is not None else datetime.now(UTC)
    start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    # Bounded wait: with a slow or hung database, queued public requests give up
    # (503 via SATimeoutError) instead of piling up threads that /ask also needs.
    if not _cache_lock.acquire(timeout=_LOCK_WAIT_SECONDS):
        raise SATimeoutError("metrics refresh already in progress")
    try:
        cached = _cache.get(engine)
        if cached is not None and cached.start == start and monotonic() < cached.expires_at:
            return start, end, cached.summary
        with engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(sa.text("SET LOCAL statement_timeout = '2s'"))
            summary = _read_metrics(conn, start, end)
        _cache[engine] = _CacheEntry(start, summary, monotonic() + _CACHE_TTL_SECONDS)
    finally:
        _cache_lock.release()
    return start, end, summary
