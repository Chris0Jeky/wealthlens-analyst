"""Lexical retrieval over Postgres full-text search.

Queries the `chunks` table's tsvector column (GIN-indexed; see
migrations/versions/0001_chunks.py) and returns ranked chunk hits. Official
statistics are full of exact terms ("decile", "nil-rate band", years, table
numbers), which is why the lexical leg of the hybrid is load-bearing.

The query uses `websearch_to_tsquery('english', ...)` so the parse matches the
'english' regconfig baked into the STORED `ts` generated column, and it accepts
human search syntax (quoted phrases, OR, -negation) without erroring on
punctuation. Ranking is `ts_rank`; ties break on `chunk_id` so the order is
deterministic (reproducible retrieval + stable RRF input, ADR 0001).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Engine

from wealthlens_analyst.config import load_settings
from wealthlens_analyst.db import engine_from_settings


@dataclass(frozen=True)
class ChunkHit:
    """One retrieved chunk with its provenance, as stored at ingestion time.

    Provenance fields mirror the chunks table columns (ADR 0001 §4):
    citations resolve from these, never reconstructed later.
    """

    chunk_id: int
    source_id: str  # registry id in registries/sources.yml
    document_id: str
    section: str | None
    page: int | None
    span: str | None
    text: str
    rank: int  # 1-based rank within this retriever's result list
    score: float  # retriever-native score (ts_rank / cosine similarity)


# websearch_to_tsquery is called once in a CTE and referenced twice (filter +
# rank) so the user query is parsed a single time. The WHERE `ts @@ q` is what
# the GIN index (chunks_ts_gin) serves; ORDER BY ts_rank then LIMIT.
_FTS_SQL = sa.text(
    """
    WITH q AS (SELECT websearch_to_tsquery('english', :query) AS query)
    SELECT c.chunk_id, c.source_id, c.document_id, c.section, c.page, c.span,
           c.text, ts_rank(c.ts, q.query) AS score
    FROM chunks AS c, q
    WHERE c.ts @@ q.query
    ORDER BY score DESC, c.chunk_id ASC
    LIMIT :limit
    """
)


def _hits_from_rows(rows: Iterable[Any]) -> list[ChunkHit]:
    """Map result rows to ranked ChunkHits (rank is 1-based row position).

    Pure and DB-free so the provenance mapping + ranking can be unit-tested
    without a database (CI has no Postgres); the SQL itself is verified live.
    """
    return [
        ChunkHit(
            chunk_id=int(row.chunk_id),
            source_id=row.source_id,
            document_id=row.document_id,
            section=row.section,
            page=row.page,
            span=row.span,
            text=row.text,
            rank=index + 1,
            score=float(row.score),
        )
        for index, row in enumerate(rows)
    ]


def search_fts(query: str, *, limit: int = 50, engine: Engine | None = None) -> list[ChunkHit]:
    """Return the top `limit` chunks by Postgres full-text rank, best first.

    Deterministic: equal ts_rank scores break on chunk_id, so the same corpus +
    query always yields the same order (a stable input to RRF fusion, H1-12).
    A negative `limit` is rejected (fail-loud, matching fuse_rrf); `limit=0` and a
    query that matches nothing both return []. `engine` defaults to one built from
    the environment (and disposed after use, since it owns its own pool); callers
    in the request path should pass a shared engine (wired in H1-13) rather than
    building one per query.
    """
    if limit < 0:
        raise ValueError(f"search_fts limit must be non-negative, got {limit}")
    if limit == 0:
        return []
    # Only dispose an engine WE created — a caller-supplied (shared) engine is the
    # caller's to manage; disposing it would tear down a pool still in use.
    created = engine is None
    if engine is None:
        engine = engine_from_settings(load_settings())
    try:
        with engine.connect() as conn:
            rows = conn.execute(_FTS_SQL, {"query": query, "limit": limit}).all()
    finally:
        if created:
            engine.dispose()
    return _hits_from_rows(rows)
