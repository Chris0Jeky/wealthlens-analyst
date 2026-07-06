"""Unit tests for the DB-free guards of dense retrieval (H1-11).

The cosine SQL and the embed/write loop need Postgres + pgvector, so they are
verified live during ingestion (analyst-db on :15432). These tests pin the
input-validation and short-circuit behaviour that must hold WITHOUT a database or
a model call.
"""

from __future__ import annotations

import pytest

import wealthlens_analyst.retrieval.dense as dense
from wealthlens_analyst.retrieval.dense import embed_corpus, search_dense, search_dense_by_vector


def test_search_dense_negative_limit_fails_loud() -> None:
    # Mirrors search_fts: a negative limit is a bug, not "no results".
    with pytest.raises(ValueError, match="non-negative"):
        search_dense("anything", limit=-1)


def test_search_dense_zero_limit_short_circuits_without_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    # limit=0 must return [] BEFORE building a client / spending on an embedding call.
    def _boom() -> object:
        raise AssertionError("get_client must not be called when limit == 0")

    monkeypatch.setattr(dense, "get_client", _boom)
    assert search_dense("anything", limit=0) == []


def test_embed_corpus_non_positive_batch_size_fails_loud() -> None:
    # The batch_size guard runs before any client/DB work, so this needs neither.
    with pytest.raises(ValueError, match="batch_size"):
        embed_corpus(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        embed_corpus(batch_size=-5)


def test_search_dense_by_vector_negative_limit_fails_loud() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        search_dense_by_vector([0.1], limit=-1)


def test_search_dense_by_vector_zero_limit_short_circuits_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    # limit=0 must return [] BEFORE building an engine (no DB in CI).
    def _boom(settings: object) -> object:
        raise AssertionError("engine_from_settings must not be called when limit == 0")

    monkeypatch.setattr(dense, "engine_from_settings", _boom)
    assert search_dense_by_vector([0.1], limit=0) == []


def test_search_dense_by_vector_wrong_dimension_fails_loud_before_the_db() -> None:
    # A misconfigured embedding model must be a clear ValueError, not a
    # database-level pgvector error (query-path counterpart of embed_corpus's
    # dim validation).
    with pytest.raises(ValueError, match="dims"):
        search_dense_by_vector([0.1, 0.2, 0.3], limit=5)
