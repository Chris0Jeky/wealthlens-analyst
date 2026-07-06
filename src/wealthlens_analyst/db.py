"""Database access for the analyst: engine factory + Core table handles.

The DDL is owned by the hand-written Alembic migrations under
``migrations/versions/`` — they stay the single source of truth (env.py diffs
against no metadata: ``target_metadata = None``). The lightweight SQLAlchemy
Core table defined here mirrors migration ``0001_chunks`` for ONE purpose:
issuing typed INSERT/DELETE statements on the ingestion write path (H1-09). It
is never used to emit DDL.

Server-managed columns are deliberately omitted, because the write path must
never supply them:

- ``chunk_id`` — an always-``Identity`` primary key (Postgres assigns it);
- ``ts`` — a STORED generated ``tsvector`` (Postgres maintains the FTS index);
- ``created_at`` — a ``now()`` server default.
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine

from wealthlens_analyst.config import Settings

#: Embedding dimensionality — mirrors migration 0002_embeddings (OpenAI
#: text-embedding-3-small, ADR 0003 D2). The vector column rejects any other
#: length, so this is the single in-code source of truth the embed path checks
#: against before insert.
EMBEDDING_DIM = 1536

# Mirrors the INSERTABLE columns of migration 0001_chunks. A standalone
# MetaData (never reflected, never create_all'd) so this can be imported with no
# database connection — CI has no Postgres service.
chunks_metadata = sa.MetaData()
chunks_table = sa.Table(
    "chunks",
    chunks_metadata,
    sa.Column("source_id", sa.Text, nullable=False),
    sa.Column("document_id", sa.Text, nullable=False),
    sa.Column("section", sa.Text),
    sa.Column("page", sa.Integer),
    sa.Column("span", sa.Text),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("token_count", sa.Integer, nullable=False),
    sa.Column("access_date", sa.Date, nullable=False),
)

# Mirrors the INSERTABLE columns of migration 0002_embeddings (chunk_id + model +
# vector). `created_at` is a now() server default and is deliberately omitted (the
# write path must never supply it), matching the chunks_table convention above.
embeddings_table = sa.Table(
    "embeddings",
    chunks_metadata,
    sa.Column("chunk_id", sa.BigInteger, primary_key=True),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
)

# Mirrors the INSERTABLE columns of migration 0004_query_log (per-request
# accounting, ADR 0002). Server-managed `query_id` (Identity) and `asked_at`
# (now() default) are deliberately omitted, matching the convention above.
# `question_sha` is a hash, never the raw question text (the migration's
# privacy rationale: the metrics endpoint is public and a query log is user
# data). `decision` values come from budget.models.QueryDecision; the DDL's
# CHECK constraint keeps the DB honest if code and schema ever drift.
query_log_table = sa.Table(
    "query_log",
    chunks_metadata,
    sa.Column("question_sha", sa.Text, nullable=False),
    sa.Column("decision", sa.Text, nullable=False),
    sa.Column("gate_signal", sa.Double),
    sa.Column("tokens_in", sa.Integer, nullable=False),
    sa.Column("tokens_out", sa.Integer, nullable=False),
    sa.Column("cost_gbp", sa.Numeric(12, 8), nullable=False),
    sa.Column("latency_ms", sa.Integer, nullable=False),
    sa.Column("rerank_used", sa.Boolean, nullable=False),
)


def engine_from_settings(settings: Settings) -> Engine:
    """Build a SQLAlchemy engine from the configured ``DATABASE_URL``.

    Fails loudly when the URL is unset: an ingest or retrieval call with no
    database configured is a setup error, not something to paper over with a
    silent default (which would surface later as a confusing connection error).

    ``pool_pre_ping``: the API holds one engine for the whole app lifetime
    (api/app.py's lifespan), so without it the first request after a database
    restart fails on a stale pooled connection even though the DB is healthy
    (verified live: /healthz false-503s and /ask 500s for exactly one request
    per blip). Costs one lightweight ping per checkout — harmless for the
    short-lived CLI/ingest callers that share this factory.

    ``connect_timeout`` bounds connection ESTABLISHMENT (libpq), so a probe
    against an unreachable host fails promptly instead of hanging for the OS
    TCP timeout. It does not bound an already-established connection that
    goes silent mid-query — acceptable for a liveness probe; pre_ping covers
    the common stale-connection case.
    """
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured (see .env.example); cannot reach the analyst database")
    return sa.create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )
