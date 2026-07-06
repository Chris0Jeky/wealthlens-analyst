"""query_log: per-request accounting (ADR 0002).

One row per /ask. Feeds BOTH spend-cap accounting (sum of cost_gbp in the
active period) and the public metrics page (p50/p95 latency, cost/query).
Stores a hash of the question, not the raw text — the metrics endpoint is
public and a query log is user data.

`decision` values match wealthlens_analyst.budget.models.QueryDecision; the
CHECK constraint keeps the DB honest if code and schema ever drift.

Revision ID: 0004_query_log
Revises: 0003_budgets
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_query_log"
down_revision = "0003_budgets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_log",
        sa.Column("query_id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column(
            "asked_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("question_sha", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("gate_signal", sa.Double(), nullable=True),
        sa.Column("tokens_in", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("tokens_out", sa.Integer(), server_default=sa.text("0"), nullable=False),
        # scale 8: a single embedding call costs fractions of a micro-pound;
        # at scale 6 those would round to zero and the summed spend (the hard
        # cap's input) would systematically understate reality.
        sa.Column("cost_gbp", sa.Numeric(12, 8), server_default=sa.text("0"), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("rerank_used", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.CheckConstraint(
            "decision IN ('answered', 'refused', 'over_budget', 'error')",
            name="query_log_decision_valid",
        ),
    )
    op.create_index("query_log_asked_at", "query_log", ["asked_at"])


def downgrade() -> None:
    op.drop_index("query_log_asked_at", table_name="query_log")
    op.drop_table("query_log")
