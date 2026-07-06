"""Optional reranking of the fused candidate list (ADR 0001 §3).

Sits behind the RERANK_ENABLED feature flag, default OFF: with the flag off,
the fused RRF order passes through untouched (byte-identical behaviour, locked
by a test in H1-16). Reranker: ADR 0003 D1 = Cohere Rerank 4 Fast, with
self-hosted bge-reranker-v2-m3 as the documented exit ramp — this module keeps
the choice behind one function so a swap is config/implementation, not a
refactor.

If the reranker is API-based, its calls are metered through the budget
middleware like every other model call (ADR 0002).

Pending: task H1-16 in tasks/hero1-backlog.md.
"""

from __future__ import annotations

from wealthlens_analyst.retrieval.fts import ChunkHit


def rerank(query: str, candidates: list[ChunkHit], *, top_k: int = 8) -> list[ChunkHit]:
    """Rerank fused candidates; return the top_k in reranked order.

    Callers must check Settings.rerank_enabled BEFORE calling — flag-off
    means this function is never invoked.
    """
    raise NotImplementedError("H1-16: reranker not yet implemented (ADR 0003 D1: Cohere Rerank 4 Fast)")
