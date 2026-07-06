"""embeddings: pgvector column + HNSW cosine index (ADR 0001).

Separate table (not a column on chunks) so re-embedding with a different
model is an insert+swap, and so the producing model is recorded per vector —
an embedding without its model id is unreproducible.

Dimension 1536 = OpenAI text-embedding-3-small (ADR 0003 D2, adopted
2026-06-11 under Chris's delegation; the recommendation in the memo). Under
pgvector's 2,000-dim HNSW limit. If D2 is overridden, a new migration swaps
the column/index — do not edit this one.

The extension is created here but NOT dropped on downgrade: dropping a shared
extension from a downgrade is a foot-gun if anything else ever uses it.

Revision ID: 0002_embeddings
Revises: 0001_chunks
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002_embeddings"
down_revision = "0001_chunks"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embeddings",
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "embeddings_hnsw",
        "embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("embeddings_hnsw", table_name="embeddings")
    op.drop_table("embeddings")
