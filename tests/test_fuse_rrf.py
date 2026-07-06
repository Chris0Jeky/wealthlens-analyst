"""Unit tests pinning the RRF fusion math (H1-12, ADR 0001).

Pure-Python: no DB, no model calls. Exercises the deterministic-check path the
eval harness relies on. Covers overlap, disjoint lists, tie-breaking, k
override, limit truncation, empty inputs, and provenance preservation.
"""

from __future__ import annotations

import pytest

from wealthlens_analyst.retrieval.fts import ChunkHit
from wealthlens_analyst.retrieval.fuse_rrf import RRF_K, fuse_rrf


def _hit(chunk_id: int, rank: int, score: float = 0.0) -> ChunkHit:
    """Build a ChunkHit with throwaway provenance; rank is retriever-native."""
    return ChunkHit(
        chunk_id=chunk_id,
        source_id="ons-was-wealth",
        document_id=f"doc-{chunk_id}",
        section=None,
        page=None,
        span=None,
        text=f"chunk {chunk_id}",
        rank=rank,
        score=score,
    )


def _ranked(*chunk_ids: int) -> list[ChunkHit]:
    """A ranked list where input order == retriever rank (1-based)."""
    return [_hit(cid, rank=i) for i, cid in enumerate(chunk_ids, start=1)]


def test_single_list_preserves_order() -> None:
    fused = fuse_rrf(_ranked(10, 20, 30), [])
    assert [h.chunk_id for h in fused] == [10, 20, 30]
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[1].score == pytest.approx(1.0 / (RRF_K + 2))


def test_overlap_boosts_shared_chunk() -> None:
    # Chunk 99 is mid-ranked in both lists; it should outscore single-list hits.
    lexical = _ranked(1, 99, 2)
    dense = _ranked(3, 99, 4)
    fused = fuse_rrf(lexical, dense)
    assert fused[0].chunk_id == 99


def test_disjoint_lists_are_deterministic() -> None:
    fused = fuse_rrf(_ranked(1, 2), _ranked(3, 4))
    # 1 and 3 share score 1/(k+1); 2 and 4 share 1/(k+2). Tie-break by chunk_id.
    assert [h.chunk_id for h in fused] == [1, 3, 2, 4]


def test_tie_break_by_chunk_id() -> None:
    # Symmetric lists give chunks 5 and 7 identical RRF scores -> lower id first.
    fused = fuse_rrf(_ranked(5, 7), _ranked(7, 5))
    assert [h.chunk_id for h in fused] == [5, 7]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_exact_rrf_math_known_case() -> None:
    # Chunk 42: lexical position 1, dense position 2, k=60.
    fused = fuse_rrf(_ranked(42, 1), _ranked(1, 42))
    score_42 = next(h.score for h in fused if h.chunk_id == 42)
    assert score_42 == pytest.approx(1.0 / 61 + 1.0 / 62)


def test_empty_inputs_return_empty() -> None:
    assert fuse_rrf([], []) == []


def test_limit_truncates_to_top_n() -> None:
    fused = fuse_rrf(_ranked(1, 2, 3, 4, 5), [], limit=2)
    assert [h.chunk_id for h in fused] == [1, 2]


def test_k_override_scales_single_list_scores() -> None:
    # Scoped to a SINGLE list: order is monotonic in position for any k>0, so
    # only the absolute scores change. (For two overlapping lists, k can also
    # reweight overlap-vs-depth and change the order — not asserted here.)
    lst = _ranked(1, 2, 3)
    default = fuse_rrf(lst, [])
    small_k = fuse_rrf(lst, [], k=1)
    assert [h.chunk_id for h in default] == [h.chunk_id for h in small_k]
    assert small_k[0].score > default[0].score


def test_returned_rank_is_fused_1based() -> None:
    fused = fuse_rrf(_ranked(10, 20, 30), [])
    assert [h.rank for h in fused] == [1, 2, 3]


def test_provenance_preserved() -> None:
    original = _hit(7, rank=1)
    fused = fuse_rrf([original], [])
    assert fused[0].source_id == original.source_id
    assert fused[0].document_id == original.document_id
    assert fused[0].text == original.text
    assert fused[0].section == original.section


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        fuse_rrf(_ranked(1), [], k=0)


def test_negative_limit_raises() -> None:
    # Fail loudly rather than silently returning a slice-truncated set
    # (Python's list[:-1] would drop from the end — a silent wrong result).
    with pytest.raises(ValueError, match="limit must be non-negative"):
        fuse_rrf(_ranked(1, 2, 3), [], limit=-1)


def test_limit_zero_returns_empty() -> None:
    assert fuse_rrf(_ranked(1, 2, 3), [], limit=0) == []
