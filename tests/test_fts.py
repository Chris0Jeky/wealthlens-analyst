"""Lock the FTS query's provenance mapping + guards (H1-10, ADR 0001).

DB-free by design (CI has no Postgres): these cover the pure row->ChunkHit
mapping, 1-based ranking, type coercion, and the limit guards. The SQL itself —
websearch_to_tsquery over the GIN-indexed tsvector, ts_rank ordering, and that
the index is used — is verified live against the analyst-db container (recorded
in the PR), not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wealthlens_analyst.retrieval.fts import ChunkHit, _hits_from_rows, search_fts


def _row(**overrides: object) -> SimpleNamespace:
    """A stand-in result row (attribute access, like a SQLAlchemy Row)."""
    fields: dict[str, object] = {
        "chunk_id": 1,
        "source_id": "hmrc-cgt-statistics",
        "document_id": "hmrc-cgt-size-of-gain",
        "section": "Capital Gains Tax by size of gain: £1,000,000 to under £2,000,000",
        "page": None,
        "span": "gain_band=£1,000,000+",
        # Deliberately distinct from `section` so a text<->section swap is caught.
        "text": "EVIDENCE BODY: taxpayers with gains of £1,000,000 and above accounted for 60.9% of gains.",
        "score": 0.5,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _ExplodingEngine:
    """Fails if a connection is opened — proves the guards short-circuit first."""

    def connect(self) -> object:
        raise AssertionError("database connection opened despite a guard that should short-circuit")


def test_hits_from_rows_maps_provenance_and_ranks() -> None:
    """Rows become ChunkHits with full provenance and a 1-based rank in row order."""
    rows = [_row(chunk_id=7, score=0.9), _row(chunk_id=3, score=0.4)]
    hits = _hits_from_rows(rows)

    assert [h.rank for h in hits] == [1, 2]  # 1-based, preserves the SQL ORDER BY
    assert [h.chunk_id for h in hits] == [7, 3]
    first = hits[0]
    assert isinstance(first, ChunkHit)
    assert first.source_id == "hmrc-cgt-statistics"
    assert first.document_id == "hmrc-cgt-size-of-gain"
    assert first.span == "gain_band=£1,000,000+"
    assert first.page is None
    # Lock section AND text (the evidence body a citation surfaces) so a
    # text<->section swap in the mapper is caught.
    assert first.section == "Capital Gains Tax by size of gain: £1,000,000 to under £2,000,000"
    assert first.text == "EVIDENCE BODY: taxpayers with gains of £1,000,000 and above accounted for 60.9% of gains."
    assert first.score == pytest.approx(0.9)


def test_hits_from_rows_empty_is_empty() -> None:
    assert _hits_from_rows([]) == []


def test_hits_from_rows_coerces_types() -> None:
    """chunk_id -> int, score -> float, even if the driver hands back other numerics."""
    (hit,) = _hits_from_rows([_row(chunk_id=True, score=1)])
    assert isinstance(hit.chunk_id, int) and hit.chunk_id == 1
    assert isinstance(hit.score, float) and hit.score == 1.0


def test_search_fts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        search_fts("capital gains", limit=-1, engine=_ExplodingEngine())  # type: ignore[arg-type]


def test_search_fts_zero_limit_returns_empty_without_db() -> None:
    """limit=0 is a valid 'top-0' request: [] without ever opening a connection."""
    assert search_fts("capital gains", limit=0, engine=_ExplodingEngine()) == []  # type: ignore[arg-type]
