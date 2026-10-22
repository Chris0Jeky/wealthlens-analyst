"""FastAPI middleware enforcing the hard spend cap (ADR 0002).

Before any model call can happen, the middleware atomically checks accumulated
spend plus in-flight reservations plus the request's worst-case bound. Over the cap ->
HTTP 429 with a structured refusal body (same schema family as the abstention
refusal, distinct reason code). Fail-closed: no budget configured -> blocked.

This behaviour is exercised by a deterministic eval check that seeds a budget,
exhausts it, and asserts the 429 shape — no model call required.

The /ask handler invokes this hook before either retrieval mode (H1-27).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from threading import Lock

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from wealthlens_analyst.budget.bounds import request_cost_bound_gbp
from wealthlens_analyst.budget.models import (
    BudgetUnavailableError,
    QueryDecision,
    active_budget_cap_gbp,
    current_spend_gbp,
    record_query,
)
from wealthlens_analyst.config import load_settings

logger = logging.getLogger(__name__)

# Stable machine-readable reason served in the 429 body.
REFUSAL_REASON_OVER_BUDGET = "over_budget"

_reservation_lock = Lock()
_in_flight_reserved = Decimal("0")


class BudgetRefusal(Exception):
    """Mapped by the app to the published top-level 429 body, without detail wrapping."""

    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__(REFUSAL_REASON_OVER_BUDGET)


@contextmanager
def budget_guard(engine: Engine, question: str, *, now: datetime | None = None) -> Iterator[None]:
    """Atomically check and reserve before spend; release after request accounting.

    Called only by /ask, after input validation and before retrieval, so free
    routes stay exempt. Both configured and persisted caps bind: the lower wins.
    The caller must keep success/refusal/error query_log writes INSIDE the context.
    The lock covers the spend read, comparison and reservation; model calls run
    unlocked. Reservations are per process, matching ADR 0003's single uvicorn
    worker deployment. They do not coordinate multiple processes or other spenders.
    """
    # ponytail: multiple workers require database-level atomic reservations.
    global _in_flight_reserved
    reserved = Decimal("0")
    read_failed = False
    try:
        settings = load_settings()
        configured = settings.budget_monthly_cap_gbp
        if configured is None:
            raise BudgetUnavailableError("No configured budget cap")
        cap = Decimal(str(configured))
        if not cap.is_finite() or cap <= 0:
            raise BudgetUnavailableError("Invalid configured budget cap")
        bound = request_cost_bound_gbp(question, settings)
        with _reservation_lock:
            now = now or datetime.now(UTC)
            with engine.connect() as conn:
                row_cap = active_budget_cap_gbp(conn, now=now)
                spend = current_spend_gbp(conn, now=now)
            if spend + _in_flight_reserved + bound <= min(cap, row_cap):
                _in_flight_reserved += bound
                reserved = bound
    except (ValueError, InvalidOperation):
        logger.warning("budget: invalid or unavailable budget; spending blocked")
    except SQLAlchemyError:
        # Never include exception text: connection errors can contain credentials.
        logger.warning("budget: database read failed; spending blocked")
        read_failed = True

    if reserved:
        try:
            yield
        finally:
            with _reservation_lock:
                _in_flight_reserved -= reserved
        return

    # Zero-spend refusals are accounted best-effort. Do not retry an unavailable
    # database just to record that it refused to authorize a model call.
    if not read_failed:
        try:
            record_query(
                engine,
                question=question,
                decision=QueryDecision.OVER_BUDGET,
                tokens_in=0,
                tokens_out=0,
                cost_gbp=0,
                latency_ms=0,
            )
        except SQLAlchemyError:
            logger.warning("budget: refusal accounting unavailable")
    raise BudgetRefusal(question)
