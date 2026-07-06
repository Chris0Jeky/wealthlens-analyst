"""SQLAlchemy models for budget enforcement and per-query accounting.

Two tables (migrations/versions/0003_budgets.py and 0004_query_log.py):

- budgets: the hard cap per period (monthly GBP). The middleware reads the
  active row; a MISSING row blocks spend rather than allowing it (fail-closed,
  ADR 0002). At most one row may be active (partial unique index), and `active`
  defaults true — so rolling a period over is deactivate-then-insert (H1-27).
- query_log: one row per /ask request — tokens in/out, cost, latency, and the
  decision (answered / refused / over_budget / error). The public metrics page
  (p50/p95 latency, cost per query) is computed from this table alone, and the
  spend cap (H1-27) sums its cost_gbp — which is why a served-but-unrecorded
  request is an accounting hole, not a cosmetic gap.

The H1-15 write path lives here: `query_log_row` (pure — the row shape) and
`record_query` (the insert). Pending: H1-27 (budget read/enforcement).
"""

from __future__ import annotations

import hashlib
import math
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum

from sqlalchemy import Engine

from wealthlens_analyst.db import query_log_table


class QueryDecision(StrEnum):
    """Terminal outcome of one /ask request, recorded in query_log.

    A successful ?debug=retrieval request records ANSWERED: the request was
    served (retrieval ran and returned candidates); its tokens_out is 0
    because no generation happened, but its embedding tokens_in/cost_gbp are
    real spend and are recorded (ADR 0002: every model call is metered).
    """

    ANSWERED = "answered"
    REFUSED = "refused"  # abstention gate: cannot answer from corpus
    OVER_BUDGET = "over_budget"  # spend cap exceeded -> 429
    ERROR = "error"


#: query_log.cost_gbp is Numeric(12, 8). Quantising app-side makes the stored
#: value exact and deterministic instead of relying on implicit driver-side
#: float coercion. The mode is ROUND_HALF_EVEN (Decimal's default); note that
#: Postgres itself rounds numeric halves AWAY FROM ZERO, which is one more
#: reason not to leave the rounding to the database.
_COST_QUANTUM = Decimal("0.00000001")


def query_log_row(
    *,
    question: str,
    decision: QueryDecision,
    tokens_in: int,
    tokens_out: int,
    cost_gbp: float,
    latency_ms: int,
    gate_signal: float | None = None,
    rerank_used: bool = False,
) -> dict[str, object]:
    """Build the query_log INSERT values for one /ask request. Pure.

    The question is stored as a SHA-256 hex digest, never raw text
    (migration 0004's privacy rationale: the metrics endpoint is public and
    a query log is user data). Negative token counts or cost are a caller
    bug and fail loudly — a negative spend would silently reduce the summed
    cap input (ADR 0002).
    """
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError(f"token counts must be non-negative, got in={tokens_in} out={tokens_out}")
    # isfinite, not just `< 0`: nan compares False to everything (so it would
    # slip past a sign check and blow up later as an opaque InvalidOperation
    # in quantize), and inf can't be stored. Mirrors config._parse_budget_cap,
    # the other seam guarding the same cap arithmetic (ADR 0002).
    if not math.isfinite(cost_gbp) or cost_gbp < 0:
        raise ValueError(f"cost_gbp must be a non-negative, finite number, got {cost_gbp}")
    if latency_ms < 0:
        raise ValueError(f"latency_ms must be non-negative, got {latency_ms}")
    return {
        "question_sha": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "decision": decision.value,
        "gate_signal": gate_signal,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_gbp": Decimal(str(cost_gbp)).quantize(_COST_QUANTUM, rounding=ROUND_HALF_EVEN),
        "latency_ms": latency_ms,
        "rerank_used": rerank_used,
    }


def record_query(
    engine: Engine,
    *,
    question: str,
    decision: QueryDecision,
    tokens_in: int,
    tokens_out: int,
    cost_gbp: float,
    latency_ms: int,
    gate_signal: float | None = None,
    rerank_used: bool = False,
) -> None:
    """Insert one query_log row (its own short transaction).

    Raises on failure — the CALLER decides what an accounting failure means
    for its response (the /ask success path fails the request rather than
    serve unmetered spend; an error path logs and preserves the original
    error). Swallowing it here would make that policy impossible.
    """
    row = query_log_row(
        question=question,
        decision=decision,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_gbp=cost_gbp,
        latency_ms=latency_ms,
        gate_signal=gate_signal,
        rerank_used=rerank_used,
    )
    with engine.begin() as conn:
        conn.execute(query_log_table.insert(), [row])


def current_spend_gbp() -> float:
    """Sum of query_log.cost_gbp within the active budget period."""
    raise NotImplementedError("H1-27: budget accounting not yet implemented")
