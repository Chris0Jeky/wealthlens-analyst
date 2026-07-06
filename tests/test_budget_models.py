"""Unit tests for the query_log row builder (H1-15).

query_log_row is the pure half of the write path — testable without Postgres
(CI has no database service). The INSERT itself (record_query) is verified
live against analyst-db, matching the house convention for SQL.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from wealthlens_analyst.budget.models import QueryDecision, query_log_row
from wealthlens_analyst.db import query_log_table


def test_row_shape_matches_the_insertable_table_columns_exactly() -> None:
    # Drift-lock: if migration 0004's mirror in db.py gains or loses a column,
    # the row builder must change in the same commit.
    row = query_log_row(
        question="q",
        decision=QueryDecision.ANSWERED,
        tokens_in=1,
        tokens_out=0,
        cost_gbp=0.0,
        latency_ms=5,
    )
    assert set(row) == {column.name for column in query_log_table.columns}


def test_question_is_stored_as_sha256_never_raw_text() -> None:
    question = "how much wealth does the top 1% hold?"
    row = query_log_row(
        question=question,
        decision=QueryDecision.ANSWERED,
        tokens_in=7,
        tokens_out=0,
        cost_gbp=2.5e-7,
        latency_ms=12,
    )
    assert row["question_sha"] == hashlib.sha256(question.encode("utf-8")).hexdigest()
    assert question not in str(row.values())


def test_row_carries_decision_value_and_quantised_cost() -> None:
    row = query_log_row(
        question="q",
        decision=QueryDecision.ERROR,
        tokens_in=7,
        tokens_out=0,
        cost_gbp=2.5e-7,
        latency_ms=12,
        gate_signal=None,
        rerank_used=False,
    )
    assert row["decision"] == "error"  # the string the CHECK constraint knows
    # Numeric(12, 8): quantised to 8 dp explicitly, not left to driver coercion.
    assert row["cost_gbp"] == Decimal("0.00000025")
    assert row["tokens_in"] == 7
    assert row["tokens_out"] == 0
    assert row["latency_ms"] == 12
    assert row["gate_signal"] is None
    assert row["rerank_used"] is False


def test_sub_quantum_cost_survives_the_scale() -> None:
    # The migration chose scale 8 precisely so a single embedding call
    # (fractions of a micro-pound) does not round to zero and silently
    # understate the summed spend the cap reads.
    row = query_log_row(
        question="q",
        decision=QueryDecision.ANSWERED,
        tokens_in=4,
        tokens_out=0,
        cost_gbp=6e-8,
        latency_ms=1,
    )
    assert row["cost_gbp"] == Decimal("0.00000006")
    assert row["cost_gbp"] > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("tokens_in", -1), ("tokens_out", -2), ("cost_gbp", -0.01), ("latency_ms", -5)],
)
def test_negative_accounting_fails_loud(field: str, value: float) -> None:
    # A negative spend or count would silently reduce the summed cap input.
    kwargs: dict[str, object] = {
        "question": "q",
        "decision": QueryDecision.ANSWERED,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_gbp": 0.0,
        "latency_ms": 0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="non-negative"):
        query_log_row(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_cost_fails_loud(bad_cost: float) -> None:
    # nan compares False to everything, so a bare `< 0` guard would pass it
    # through to an opaque InvalidOperation in quantize; inf cannot be stored.
    with pytest.raises(ValueError, match="finite"):
        query_log_row(
            question="q",
            decision=QueryDecision.ANSWERED,
            tokens_in=0,
            tokens_out=0,
            cost_gbp=bad_cost,
            latency_ms=0,
        )


def test_cost_is_quantised_to_scale_8_with_bankers_rounding() -> None:
    # Inputs that REQUIRE quantisation (exact halves at the 9th decimal), so
    # deleting the quantize call, changing the rounding mode, or mis-scaling
    # the quantum each break at least one assertion. ROUND_HALF_EVEN:
    # 1.5e-8 rounds UP to the even digit 2; 2.5e-8 rounds DOWN to the same
    # even digit 2 (half-up would give 3).
    def cost_of(value: float) -> Decimal:
        return query_log_row(
            question="q",
            decision=QueryDecision.ANSWERED,
            tokens_in=0,
            tokens_out=0,
            cost_gbp=value,
            latency_ms=0,
        )["cost_gbp"]  # type: ignore[return-value]

    assert cost_of(1.5e-8) == Decimal("0.00000002")
    assert cost_of(2.5e-8) == Decimal("0.00000002")
    assert cost_of(1.5e-8).as_tuple().exponent == -8
