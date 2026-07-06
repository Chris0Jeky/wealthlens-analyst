"""FastAPI middleware enforcing the hard spend cap (ADR 0002).

Before any model call can happen, the middleware compares accumulated spend
(budget.models.current_spend_gbp) against the active budget. Over the cap ->
HTTP 429 with a structured refusal body (same schema family as the abstention
refusal, distinct reason code). Fail-closed: no budget configured -> blocked.

This behaviour is exercised by a deterministic eval check that seeds a budget,
exhausts it, and asserts the 429 shape — no model call required.

Pending: task H1-27 in tasks/hero1-backlog.md.
"""

from __future__ import annotations

# Stable machine-readable reason served in the 429 body.
REFUSAL_REASON_OVER_BUDGET = "over_budget"


def budget_guard() -> None:
    """FastAPI dependency/middleware hook; raises the 429 when over cap.

    Wired into the /ask route in H1-27 (as a dependency rather than global
    middleware so /healthz and the metrics endpoint stay cap-exempt — they
    cost nothing).
    """
    raise NotImplementedError("H1-27: budget enforcement not yet implemented")
