"""chunks: corpus chunks with ingestion-time provenance (ADR 0001 §4).

Provenance columns are NOT NULL where universally applicable (source_id,
document_id) and nullable where source-type-specific (section, page, span) —
tabular sources have no pages, documents always have them. The PER-TYPE
requirement (tabular: section+span; document: page+span) is enforced at
ingestion (backlog H1-09), not in DDL, so the rule lives in one place.

`ts` is a STORED generated column: to_tsvector with an explicit regconfig is
immutable, so Postgres maintains the FTS index for us — ingestion cannot
forget to update it.

Revision ID: 0001_chunks
Revises: None
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_chunks"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NB: the INSERTABLE columns below are hand-mirrored by db.chunks_table (the
    # ingestion write path, H1-09) — CI has no Postgres to reflect against. Keep
    # the two in sync; a unit test couples db.chunks_table to the write path's
    # row projection so they cannot drift apart silently.
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        # registries/sources.yml id — the citation root
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("span", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("access_date", sa.Date(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "ts",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
        sa.CheckConstraint("token_count >= 0", name="chunks_token_count_nonnegative"),
    )
    op.create_index("chunks_ts_gin", "chunks", ["ts"], postgresql_using="gin")
    op.create_index("chunks_source", "chunks", ["source_id", "document_id"])


def downgrade() -> None:
    op.drop_index("chunks_source", table_name="chunks")
    op.drop_index("chunks_ts_gin", table_name="chunks")
    op.drop_table("chunks")
