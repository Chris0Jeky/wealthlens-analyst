"""Unit tests for the DB-free guards of dense retrieval (H1-11).

The cosine SQL and the embed/write loop need Postgres + pgvector, so they are
verified live during ingestion (analyst-db on :15432). These tests pin the
input-validation and short-circuit behaviour that must hold WITHOUT a database or
a model call.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine

import wealthlens_analyst.retrieval.dense as dense
from wealthlens_analyst.llm.client import EmbeddingResult, EmbeddingValidationError, LLMClient
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


def test_embed_corpus_client_error_inserts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock(spec=Engine)
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.all.return_value = [
        SimpleNamespace(chunk_id=1, text="alpha"),
        SimpleNamespace(chunk_id=2, text="beta"),
    ]
    client = MagicMock(spec=LLMClient)
    client.embed.side_effect = ValueError("OpenAI embedding response has invalid indices")
    monkeypatch.setattr(dense, "get_client", lambda: client)

    with pytest.raises(ValueError, match="indices"):
        embed_corpus(engine=engine)

    client.embed.assert_called_once_with(["alpha", "beta"])
    conn.execute.assert_called_once_with(dense._UNEMBEDDED_SQL)
    engine.begin.assert_not_called()


def test_search_dense_logs_rejected_embedding_spend(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    engine = MagicMock(spec=Engine)
    client = MagicMock(spec=LLMClient)
    rejected = EmbeddingResult(vectors=[], model="rejected-model", tokens_in=23, cost_gbp=0.00000123)
    error = EmbeddingValidationError("invalid indices", rejected)
    client.embed.side_effect = error
    monkeypatch.setattr(dense, "get_client", lambda: client)
    search = MagicMock()
    monkeypatch.setattr(dense, "search_dense_by_vector", search)

    with caplog.at_level(logging.WARNING, logger=dense.__name__), pytest.raises(EmbeddingValidationError) as raised:
        search_dense("query", engine=engine)

    assert raised.value is error
    client.embed.assert_called_once_with(["query"])
    search.assert_not_called()
    assert caplog.record_tuples == [
        (
            dense.__name__,
            logging.WARNING,
            "search_dense: query embedding validation failed for rejected-model tokens_in=23 est. cost GBP 0.00000123",
        )
    ]


@pytest.mark.parametrize("successful_batches", [0, 1])
def test_embed_corpus_logs_total_spend_on_rejected_embedding(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, successful_batches: int
) -> None:
    engine = MagicMock(spec=Engine)
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.all.return_value = [
        SimpleNamespace(chunk_id=1, text="alpha"),
        SimpleNamespace(chunk_id=2, text="beta"),
        SimpleNamespace(chunk_id=3, text="gamma"),
    ]
    insert_conn = engine.begin.return_value.__enter__.return_value
    insert_conn.execute.return_value.rowcount = 1
    client = MagicMock(spec=LLMClient)
    accepted = EmbeddingResult(
        vectors=[[0.1] * dense.EMBEDDING_DIM], model="accepted-model", tokens_in=12, cost_gbp=0.01234
    )
    rejected = EmbeddingResult(vectors=[], model="rejected-model", tokens_in=23, cost_gbp=0.02345)
    error = EmbeddingValidationError("invalid indices", rejected)
    client.embed.side_effect = [accepted] * successful_batches + [error]
    monkeypatch.setattr(dense, "get_client", lambda: client)

    with caplog.at_level(logging.WARNING, logger=dense.__name__), pytest.raises(EmbeddingValidationError) as raised:
        embed_corpus(batch_size=1, engine=engine)

    assert raised.value is error
    assert client.embed.call_count == successful_batches + 1
    assert engine.begin.call_count == successful_batches
    assert insert_conn.execute.call_count == successful_batches
    if successful_batches:
        assert insert_conn.execute.call_args.args[1] == [
            {"chunk_id": 1, "model": accepted.model, "embedding": accepted.vectors[0]}
        ]
    total = successful_batches * accepted.cost_gbp + rejected.cost_gbp
    assert caplog.record_tuples == [
        (
            dense.__name__,
            logging.WARNING,
            "embed_corpus: embedding validation failed for rejected-model tokens_in=23 est. cost GBP 0.02345000; "
            f"embedded {successful_batches} chunk(s); total est. cost GBP {total:.8f}",
        )
    ]


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
