"""Dense retrieval over pgvector.

Embeds the query through the LLM client seam (llm/client.py — the ONLY module
allowed to touch provider SDKs, ADR 0002) and runs cosine-similarity search
against the `embeddings` table (HNSW index; see
migrations/versions/0002_embeddings.py).

Embedding model: ADR 0003 D2 = OpenAI text-embedding-3-small (1536 dims,
matching migration 0002). This module reads it from configuration via the seam
and must not hard-code a provider.

Task H1-11 in tasks/hero1-backlog.md.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

from wealthlens_analyst.config import load_settings
from wealthlens_analyst.db import EMBEDDING_DIM, embeddings_table, engine_from_settings
from wealthlens_analyst.llm.client import get_client

# Reuse the FTS module's pure provenance->ChunkHit mapper: dense rows carry the
# identical columns (provenance + a score), and the mapping/ranking it performs is
# the same for both retrievers (that is the point of a shared ChunkHit). Keeping one
# tested mapper avoids two drifting copies of the citation-critical provenance map.
from wealthlens_analyst.retrieval.fts import ChunkHit, _hits_from_rows

logger = logging.getLogger(__name__)

# Chunks with no embedding row yet (anti-join). Ordered by chunk_id so batching is
# deterministic and a re-run picks up exactly the rows a previous run missed.
_UNEMBEDDED_SQL = sa.text(
    """
    SELECT c.chunk_id, c.text
    FROM chunks AS c
    LEFT JOIN embeddings AS e ON e.chunk_id = c.chunk_id
    WHERE e.chunk_id IS NULL
    ORDER BY c.chunk_id ASC
    """
)

# Cosine search. `<=>` is pgvector cosine DISTANCE (0 = identical), served by the
# HNSW index (embeddings_hnsw, vector_cosine_ops). score = 1 - distance so higher is
# better — consistent with ts_rank in fts.py, a stable RRF input (H1-12). Ties break
# on chunk_id for deterministic ordering. The query vector binds as a typed Vector
# (referenced twice but bound once).
_DENSE_SQL = sa.text(
    """
    SELECT c.chunk_id, c.source_id, c.document_id, c.section, c.page, c.span, c.text,
           1 - (e.embedding <=> :query_vec) AS score
    FROM embeddings AS e
    JOIN chunks AS c ON c.chunk_id = e.chunk_id
    ORDER BY e.embedding <=> :query_vec ASC, c.chunk_id ASC
    LIMIT :limit
    """
).bindparams(sa.bindparam("query_vec", type_=Vector(EMBEDDING_DIM)))


def search_dense_by_vector(query_vec: list[float], *, limit: int = 50, engine: Engine | None = None) -> list[ChunkHit]:
    """Cosine-search the embeddings table with an ALREADY-EMBEDDED query vector.

    The pure-DB half of dense retrieval: no model call, no spend. Callers who
    need per-request accounting (the /ask path, H1-15) embed through the client
    seam themselves and pass the vector here; `search_dense` below remains the
    one-call convenience for CLI/spot-check use. score = 1 - cosine_distance
    (higher is better); equal scores break on chunk_id for determinism (a stable
    input to RRF fusion, H1-12). A negative `limit` is rejected (fail-loud,
    matching search_fts); `limit=0` returns []. `engine` defaults to one built
    from the environment and disposed after use (it owns its pool).
    """
    if limit < 0:
        raise ValueError(f"search_dense_by_vector limit must be non-negative, got {limit}")
    if limit == 0:
        return []
    # Fail fast on a wrong-dimension vector (a misconfigured embedding model),
    # with a clear error instead of a database-level pgvector failure — the
    # query-path counterpart of embed_corpus's per-vector dim validation.
    if len(query_vec) != EMBEDDING_DIM:
        raise ValueError(
            f"query_vec has {len(query_vec)} dims, expected {EMBEDDING_DIM} (embedding model does not match migration 0002)"
        )
    created = engine is None
    if engine is None:
        engine = engine_from_settings(load_settings())
    try:
        with engine.connect() as conn:
            rows = conn.execute(_DENSE_SQL, {"query_vec": query_vec, "limit": limit}).all()
    finally:
        if created:
            engine.dispose()
    return _hits_from_rows(rows)


def search_dense(query: str, *, limit: int = 50, engine: Engine | None = None) -> list[ChunkHit]:
    """Return the top `limit` chunks by cosine similarity to `query`, best first.

    Embeds the query through the client seam, then delegates the cosine search
    to `search_dense_by_vector`. A negative `limit` is rejected (fail-loud,
    matching search_fts); `limit=0` returns [] without making an embedding call.
    `engine` defaults to one built from the environment and disposed after use
    (it owns its pool); callers pass a shared engine rather than building one
    per query. The embedding's accounting (model, tokens, cost) is logged here
    (visible cost is a product goal); the /ask request path instead embeds
    directly and persists the accounting to query_log (H1-15).
    """
    if limit < 0:
        raise ValueError(f"search_dense limit must be non-negative, got {limit}")
    if limit == 0:
        return []
    # Only dispose an engine WE created — a caller-supplied (shared) engine is the
    # caller's to manage (mirrors retrieval/fts.py::search_fts). Build it BEFORE the
    # paid embedding call so a missing DATABASE_URL fails fast (and free) rather than
    # after a wasted embed.
    created = engine is None
    if engine is None:
        engine = engine_from_settings(load_settings())
    try:
        embedded = get_client().embed([query])
        logger.info(
            "search_dense: query embedding %s tokens_in=%d est. cost GBP %.8f",
            embedded.model,
            embedded.tokens_in,
            embedded.cost_gbp,
        )
        return search_dense_by_vector(embedded.vectors[0], limit=limit, engine=engine)
    finally:
        if created:
            engine.dispose()


def embed_corpus(*, batch_size: int = 64, engine: Engine | None = None) -> int:
    """Embed every chunk lacking an embedding; return the count newly embedded.

    The batch ingestion step behind `make ingest-slice` (gated on OPENAI_API_KEY by
    the caller). Idempotent: only chunks WITHOUT an embedding row are selected, and
    the insert is ON CONFLICT DO NOTHING, so a re-run is a no-op and a partial/failed
    run resumes cleanly. Each batch is one metered embedding call; the total estimated
    GBP cost is logged (visible cost is a product goal). The per-/ask spend cap is a
    separate request-path concern (H1-15/H1-27) — this is a one-time ingestion cost.

    Fails loud if the provider returns the wrong number of vectors, or a vector whose
    dimension is not EMBEDDING_DIM (a misconfigured model must not write garbage that
    the HNSW index would then serve as a "neighbour").
    """
    if batch_size <= 0:
        raise ValueError(f"embed_corpus batch_size must be positive, got {batch_size}")
    settings = load_settings()
    client = get_client()
    created = engine is None
    if engine is None:
        engine = engine_from_settings(settings)
    try:
        with engine.connect() as conn:
            pending = conn.execute(_UNEMBEDDED_SQL).all()
        if not pending:
            logger.info("embed_corpus: every chunk already has an embedding (0 to do)")
            return 0
        logger.info("embed_corpus: %d chunk(s) to embed in batches of %d", len(pending), batch_size)
        embedded = 0
        total_cost_gbp = 0.0
        model = ""
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            # Embed OUTSIDE any transaction so no DB transaction is held open across the
            # network call; then write that batch in its OWN short transaction. A failure
            # part-way leaves earlier batches COMMITTED, and the anti-join above means a
            # re-run resumes from exactly the chunks still unembedded.
            result = client.embed([row.text for row in batch])
            model = result.model
            if len(result.vectors) != len(batch):
                raise RuntimeError(
                    f"embedding provider returned {len(result.vectors)} vectors for {len(batch)} input texts"
                )
            params: list[dict[str, object]] = []
            for row, vector in zip(batch, result.vectors, strict=True):
                if len(vector) != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"embedding for chunk {row.chunk_id} has {len(vector)} dims, expected "
                        f"{EMBEDDING_DIM} (model {result.model!r} does not match migration 0002)"
                    )
                params.append({"chunk_id": int(row.chunk_id), "model": result.model, "embedding": vector})
            with engine.begin() as conn:
                inserted = conn.execute(
                    pg_insert(embeddings_table).on_conflict_do_nothing(index_elements=["chunk_id"]),
                    params,
                ).rowcount
            # rowcount is the rows actually inserted — ON CONFLICT DO NOTHING skips a
            # chunk a concurrent run already embedded, so this never overstates. Fall
            # back to the batch size only if the driver reports rowcount unknown (-1).
            embedded += inserted if inserted >= 0 else len(params)
            total_cost_gbp += result.cost_gbp
        logger.info(
            "embed_corpus: embedded %d chunk(s) with %s; est. cost GBP %.5f",
            embedded,
            model,
            total_cost_gbp,
        )
        return embedded
    finally:
        if created:
            engine.dispose()
