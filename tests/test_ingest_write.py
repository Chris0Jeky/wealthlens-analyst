"""Lock the ingestion integrity gate + write path (H1-09, ADR 0001 §4).

Pure-Python and DB-free by design: CI has no Postgres service, so these tests
exercise the provenance gate and the fail-closed ordering of write_chunks
against a stand-in engine that explodes if a connection is ever opened. The live
write + FTS path is verified locally against the analyst-db container (recorded
in the PR), not here.

Headline guarantee (the H1-09 done-when): a deliberately provenance-stripped
chunk fails ingestion before anything is written.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from wealthlens_analyst.db import chunks_table
from wealthlens_analyst.ingest import slice_corpus
from wealthlens_analyst.ingest.slice_corpus import (
    TABLE_SPECS,
    Chunk,
    ProvenanceError,
    _chunk_to_row,
    ingest_slice,
    render_table_chunks,
    validate_chunk_provenance,
    write_chunks,
)

_WAS = next(spec for spec in TABLE_SPECS if spec.source_id == "ons-was-wealth")


def _tabular_chunk(**overrides: object) -> Chunk:
    """A conformant tabular chunk (section + span, no page); override one field."""
    fields: dict[str, object] = {
        "source_id": "ons-was-wealth",
        "document_id": "was-total-wealth-by-decile",
        "section": "Total household wealth by decile, Great Britain: 5th",
        "page": None,
        "span": "decile=5th",
        "text": "Total household wealth by decile (ONS WAS). Wealth decile 5th: aggregate total net wealth £652.0bn.",
        "token_count": 14,
        "access_date": date(2026, 5, 30),
    }
    fields.update(overrides)
    return Chunk(**fields)  # type: ignore[arg-type]  # tests inject invalid values on purpose


def _document_chunk(**overrides: object) -> Chunk:
    """A conformant document chunk (page + span; section optional per the rule)."""
    fields: dict[str, object] = {
        "source_id": "rf-intergenerational-audit-2024",
        "document_id": "rf-audit-2024.pdf",
        "section": "Executive summary",
        "page": 7,
        "span": "p7:¶2",
        "text": "Wealth has grown far faster than incomes since the 1980s.",
        "token_count": 10,
        "access_date": date(2026, 6, 1),
    }
    fields.update(overrides)
    return Chunk(**fields)  # type: ignore[arg-type]


class _ExplodingEngine:
    """Stand-in engine that fails if any connection is opened.

    Proves write_chunks validates the whole batch BEFORE touching the database:
    on an invalid chunk it must raise ProvenanceError, never reach begin().
    """

    def begin(self) -> object:
        raise AssertionError("database connection opened despite invalid/empty input")


class _DisposeTrackingEngine:
    """Stand-in engine that records dispose() calls (and explodes if used).

    Lets the DB-free tests assert ingest_slice tears down an engine it created,
    via the no-CSV error path (which raises before any begin()).
    """

    def __init__(self) -> None:
        self.disposed = 0

    def begin(self) -> object:
        raise AssertionError("database connection opened despite invalid/empty input")

    def dispose(self) -> None:
        self.disposed += 1


class _RecordingConn:
    """Records the SQL statement TYPES executed (Delete / Insert), no real DB."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, statement: object, *args: object, **kwargs: object) -> object:
        self.executed.append(type(statement).__name__)

        class _Result:
            rowcount = 0

        return _Result()

    def __enter__(self) -> _RecordingConn:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _RecordingEngine:
    """Stand-in engine that records the statements write_chunks issues."""

    def __init__(self) -> None:
        self.conn = _RecordingConn()

    def begin(self) -> _RecordingConn:
        return self.conn


# --- the gate accepts conformant chunks ---------------------------------------


def test_gate_accepts_conformant_tabular_chunk() -> None:
    validate_chunk_provenance(_tabular_chunk())  # no raise


def test_gate_accepts_conformant_document_chunk_without_section() -> None:
    """A document chunk needs page + span; section is optional (the per-type rule)."""
    validate_chunk_provenance(_document_chunk(section=None))  # no raise


def test_every_rendered_tabular_chunk_passes_the_gate() -> None:
    """Real renderer output (H1-07) always satisfies the gate it feeds."""
    rows = [
        {"decile": "1st (poorest)", "total_wealth_bn": "13.9"},
        {"decile": "10th (richest)", "total_wealth_bn": "5523.2"},
    ]
    rendered = render_table_chunks(rows, _WAS)
    assert rendered  # guard against a vacuous pass
    for chunk in rendered:
        validate_chunk_provenance(chunk)


# --- the gate rejects missing provenance --------------------------------------


def test_provenance_stripped_chunk_fails_ingestion() -> None:
    """The H1-09 done-when: section AND page both stripped -> uncitable -> rejected."""
    stripped = _tabular_chunk(section=None, page=None)
    with pytest.raises(ProvenanceError, match="section"):
        validate_chunk_provenance(stripped)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"source_id": ""}, "source_id"),
        ({"source_id": "   "}, "source_id"),
        ({"document_id": ""}, "document_id"),
        ({"document_id": None}, "document_id"),
        ({"span": None}, "span"),
        ({"span": ""}, "span"),
        ({"text": ""}, "text"),
        ({"text": "   "}, "text"),
        ({"section": None, "page": None}, "section"),  # provenance-stripped tabular
    ],
)
def test_gate_rejects_missing_provenance(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ProvenanceError, match=match):
        validate_chunk_provenance(_tabular_chunk(**overrides))


@pytest.mark.parametrize("page", [0, -1])
def test_gate_rejects_non_positive_document_page(page: int) -> None:
    with pytest.raises(ProvenanceError, match="page"):
        validate_chunk_provenance(_document_chunk(page=page))


# --- write_chunks: fail-closed ordering + row projection ----------------------


def test_write_chunks_validates_before_touching_the_db() -> None:
    """One invalid chunk aborts the batch before any connection is opened."""
    good = _tabular_chunk()
    bad = _tabular_chunk(span=None)
    with pytest.raises(ProvenanceError, match="span"):
        write_chunks([good, bad], engine=_ExplodingEngine())  # type: ignore[arg-type]


def test_write_chunks_empty_is_a_noop_without_db() -> None:
    """An empty slice with nothing to refresh writes nothing and never connects."""
    assert write_chunks([], engine=_ExplodingEngine()) == 0  # type: ignore[arg-type]


def test_write_chunks_replaces_then_inserts() -> None:
    """The default path deletes the document's stale rows, then inserts the batch."""
    engine = _RecordingEngine()
    written = write_chunks([_tabular_chunk()], engine=engine)  # type: ignore[arg-type]
    assert written == 1
    assert engine.conn.executed == ["Delete", "Insert"]


def test_write_chunks_no_replace_skips_the_delete() -> None:
    engine = _RecordingEngine()
    written = write_chunks([_tabular_chunk()], engine=engine, replace=False)  # type: ignore[arg-type]
    assert written == 1
    assert engine.conn.executed == ["Insert"]


def test_refresh_documents_prunes_a_source_that_rendered_zero_chunks() -> None:
    """A document that yields no chunks this run is still pruned (no stale citations).

    Models a source that reverted to illustrative/empty: write_chunks must DELETE
    its prior rows even though there is nothing to insert for it.
    """
    engine = _RecordingEngine()
    written = write_chunks(
        [],
        engine=engine,  # type: ignore[arg-type]
        refresh_documents={("hmrc-tax-receipts", "hmrc-tax-nic-receipts")},
    )
    assert written == 0
    assert engine.conn.executed == ["Delete"]  # pruned the stale document, inserted nothing


def test_ingest_slice_raises_when_no_source_csvs_present(tmp_path: Path) -> None:
    """A fresh checkout with no processed inputs fails loudly, not silently empty."""
    with pytest.raises(RuntimeError, match="no processed source CSVs"):
        ingest_slice(engine=_ExplodingEngine(), processed_dir=tmp_path)  # type: ignore[arg-type]


def test_ingest_slice_disposes_an_engine_it_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest_slice tears down an engine it built itself (mirrors search_fts).

    Uses the no-CSV error path (empty dir) so no DB is needed: even when ingestion
    raises, the internally-created connection pool must be disposed in the finally.
    """
    created_engine = _DisposeTrackingEngine()
    monkeypatch.setattr(slice_corpus, "engine_from_settings", lambda _settings: created_engine)
    monkeypatch.setattr(slice_corpus, "load_settings", lambda: None)
    with pytest.raises(RuntimeError, match="no processed source CSVs"):
        ingest_slice(processed_dir=tmp_path)  # engine=None -> built here -> must dispose
    assert created_engine.disposed == 1


def test_ingest_slice_does_not_dispose_a_caller_supplied_engine(tmp_path: Path) -> None:
    """A shared/caller-supplied engine is the caller's to manage — never disposed here."""
    engine = _DisposeTrackingEngine()
    with pytest.raises(RuntimeError, match="no processed source CSVs"):
        ingest_slice(engine=engine, processed_dir=tmp_path)  # type: ignore[arg-type]
    assert engine.disposed == 0


def test_ingest_slice_disposes_its_engine_if_dir_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure resolving the processed dir (after the engine is built) still disposes.

    Guards the edge where _processed_dir_default() raises between engine creation and
    the main body — the self-created engine must not leak (caught in PR #456 review).
    """
    created_engine = _DisposeTrackingEngine()
    monkeypatch.setattr(slice_corpus, "engine_from_settings", lambda _settings: created_engine)
    monkeypatch.setattr(slice_corpus, "load_settings", lambda: None)

    def _boom() -> Path:
        raise RuntimeError("cannot locate repo root")

    monkeypatch.setattr(slice_corpus, "_processed_dir_default", _boom)
    with pytest.raises(RuntimeError, match="cannot locate repo root"):
        ingest_slice()  # engine=None + processed_dir=None -> dir resolution raises inside try
    assert created_engine.disposed == 1


def test_chunk_row_omits_server_managed_columns() -> None:
    """The INSERT row carries exactly the writable columns — not the server ones."""
    row = _chunk_to_row(_tabular_chunk())
    assert set(row) == {
        "source_id",
        "document_id",
        "section",
        "page",
        "span",
        "text",
        "token_count",
        "access_date",
    }
    # chunk_id (Identity), ts (generated), created_at (server default) are never supplied.
    assert "chunk_id" not in row
    assert "ts" not in row
    assert "created_at" not in row


def test_chunks_table_mirror_matches_the_row_projection() -> None:
    """db.chunks_table and _chunk_to_row are hand-maintained mirrors of migration
    0001_chunks' insertable columns (CI has no Postgres to reflect against). Couple
    them so they cannot drift apart silently — a schema change must move both."""
    table_columns = {column.name for column in chunks_table.columns}
    assert table_columns == set(_chunk_to_row(_tabular_chunk()))
    # Server-managed columns are excluded from the write path on both sides.
    assert {"chunk_id", "ts", "created_at"}.isdisjoint(table_columns)
