"""budgets: the hard spend cap (ADR 0002).

A missing row blocks spend (fail-closed — enforced in code, backlog H1-27);
the schema makes "no budget" representable and distinct from "budget of
zero". The partial unique index allows at most one active budget at a time
while keeping the full period history.

Revision ID: 0003_budgets
Revises: 0002_embeddings
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_budgets"
down_revision = "0002_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "budgets",
        sa.Column("budget_id", sa.Integer(), sa.Identity(always=True), primary_key=True),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("cap_gbp", sa.Numeric(8, 2), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("cap_gbp >= 0", name="budgets_cap_nonnegative"),
        sa.CheckConstraint("period_end > period_start", name="budgets_period_valid"),
    )
    op.create_index(
        "budgets_one_active",
        "budgets",
        ["active"],
        unique=True,
        postgresql_where=sa.text("active"),
    )


def downgrade() -> None:
    op.drop_index("budgets_one_active", table_name="budgets")
    op.drop_table("budgets")
