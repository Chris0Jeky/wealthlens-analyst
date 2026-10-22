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

import sqlalchemy as sa
from sqlalchemy import Engine

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
    counts = dict.fromkeys(QueryDecision, 0)
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


def read_monthly_metrics(engine: Engine, *, now: datetime | None = None) -> tuple[datetime, datetime, QueryMetrics]:
    """Read this UTC month's accounting; database failures propagate to the route."""
    now = now if now is not None else datetime.now(UTC)
    start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(_query_log.c.decision, _query_log.c.latency_ms, _query_log.c.cost_gbp).where(
                _query_log.c.asked_at >= start,
                _query_log.c.asked_at < end,
            )
        ).mappings()
        summary = summarize_queries(
            QueryMetric(QueryDecision(row["decision"]), row["latency_ms"], row["cost_gbp"]) for row in rows
        )
    return start, end, summary
