"""FastAPI middleware enforcing the hard spend cap (ADR 0002).

Before any model call can happen, the middleware compares accumulated spend
(budget.models.current_spend_gbp) against the active budget. Over the cap ->
HTTP 429 with a structured refusal body (same schema family as the abstention
refusal, distinct reason code). Fail-closed: no budget configured -> blocked.

This behaviour is exercised by a deterministic eval check that seeds a budget,
exhausts it, and asserts the 429 shape — no model call required.

The /ask handler invokes this hook before either retrieval mode (H1-27).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

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


class BudgetRefusal(Exception):
    """Mapped by the app to the published top-level 429 body, without detail wrapping."""

    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__(REFUSAL_REASON_OVER_BUDGET)


def budget_guard(engine: Engine, question: str, *, now: datetime | None = None) -> None:
    """Check before spend; missing config/row or a failed read blocks the request.

    Called only by /ask, after input validation and before retrieval, so free
    routes stay exempt. Both configured and persisted caps bind: the lower wins.
    This checks recorded spend, without reserving unknown future request cost.
    """
    now = now or datetime.now(UTC)
    read_failed = False
    try:
        configured = load_settings().budget_monthly_cap_gbp
        if configured is None:
            raise BudgetUnavailableError("No configured budget cap")
        cap = Decimal(str(configured))
        if not cap.is_finite() or cap <= 0:
            raise BudgetUnavailableError("Invalid configured budget cap")
        with engine.connect() as conn:
            row_cap = active_budget_cap_gbp(conn, now=now)
            spend = current_spend_gbp(conn, now=now)
        if spend < min(cap, row_cap):
            return
    except (ValueError, InvalidOperation):
        logger.warning("budget: invalid or unavailable budget; spending blocked")
    except SQLAlchemyError:
        # Never include exception text: connection errors can contain credentials.
        logger.warning("budget: database read failed; spending blocked")
        read_failed = True

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
