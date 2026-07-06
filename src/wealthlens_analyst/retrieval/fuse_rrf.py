"""Reciprocal Rank Fusion of the lexical and dense result lists (ADR 0001).

RRF is rank-based — score(d) = sum over lists of 1 / (k + rank(d)) — so it
needs no calibration between incomparable score distributions (ts_rank vs
cosine similarity). k = 60 is the standard constant from the original paper
(Cormack, Clarke & Buettcher 2009) and is locked by ADR 0001.

This module is pure (no DB, no model calls) so the eval harness's
deterministic checks can exercise it directly.
"""

from __future__ import annotations

from dataclasses import replace

from wealthlens_analyst.retrieval.fts import ChunkHit

RRF_K = 60


def fuse_rrf(
    lexical: list[ChunkHit],
    dense: list[ChunkHit],
    *,
    k: int = RRF_K,
    limit: int = 20,
) -> list[ChunkHit]:
    """Fuse two ranked lists into one, ordered by descending RRF score.

    RRF score per chunk is ``sum over lists of 1 / (k + rank)`` (ADR 0001).
    ``rank`` here is the chunk's 1-based POSITION in the list (``enumerate``),
    NOT the ``ChunkHit.rank`` field — the retriever-native ``ChunkHit.score``
    and ``ChunkHit.rank`` are deliberately ignored, so RRF stays robust to the
    incomparable ts_rank vs cosine score distributions.

    Preconditions (both guaranteed by the FTS/dense producers, which emit
    ``ORDER BY ... LIMIT`` so position == rank and rows are unique):
      * each input list is ordered by its retriever's rank (best first);
      * each ``chunk_id`` appears at most once per list — a duplicate within a
        single list would double-count its contribution (the chunks-table PK
        makes that unreachable in practice, so it is asserted by convention).

    Each returned hit keeps its provenance (source_id/document_id/section/
    page/span/text); its ``rank`` is reassigned to the fused 1-based rank and
    its ``score`` to the fused RRF score. Component (per-retriever) ranks are
    surfaced separately by the /ask?debug=retrieval response (H1-13), which
    still holds the raw input lists — so recall analysis can attribute a hit
    to a retriever even though this function overwrites rank/score.

    Ties break deterministically by ascending ``chunk_id`` so results are
    reproducible. ``limit`` must be >= 0. Pure function: no DB, no model calls.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    if limit < 0:
        raise ValueError(f"RRF limit must be non-negative, got {limit}")

    scores: dict[int, float] = {}
    representative: dict[int, ChunkHit] = {}
    for ranked in (lexical, dense):
        for position, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + position)
            # First occurrence wins; both inputs are deterministic, so this is
            # stable and preserves the chunk's ingestion-time provenance.
            representative.setdefault(hit.chunk_id, hit)

    ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [
        replace(representative[cid], rank=fused_rank, score=scores[cid])
        for fused_rank, cid in enumerate(ordered_ids[:limit], start=1)
    ]
