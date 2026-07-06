"""Unit tests for citation resolution (H1-19).

DB-free by design (CI has no Postgres): these cover the pure resolution logic
(fabricated / missing / unknown-source stripping, order preservation, and full
provenance + registry name/URL on resolved citations), the row->provenance
mapper, the registry loader's fail-loud guards, and that resolve_citations
short-circuits without opening a connection when no lookup is needed. The SQL
itself and the end-to-end retrieve->compose->resolve chain are verified live
against the analyst-db container (evals/checks/check_citations_live.py), not
here.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import wealthlens_analyst.answer.citations as citations
from wealthlens_analyst.answer.citations import (
    ChunkProvenance,
    Citation,
    ResolvedCitations,
    SourceMeta,
    _provenance_from_rows,
    _resolve_citations,
    load_source_registry,
    resolve_citations,
)
from wealthlens_analyst.answer.compose import ComposedAnswer

_REGISTRY = {
    "ons-was-wealth": SourceMeta(name="ONS Wealth and Assets Survey (WAS)", url="https://ons.example/was"),
    "hmrc-cgt-statistics": SourceMeta(name="HMRC Capital Gains Tax Statistics", url="https://hmrc.example/cgt"),
}
_ACCESS_DATE = date(2026, 5, 16)


class _ExplodingEngine:
    """Fails if a connection is opened — proves a lookup was short-circuited."""

    def connect(self) -> object:
        raise AssertionError("database connection opened despite no lookup being needed")


class _FakeResult:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def all(self) -> list[SimpleNamespace]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, statement: object, params: object) -> _FakeResult:
        return _FakeResult(self._rows)


class _FakeEngine:
    """Returns canned rows and records whether it was disposed."""

    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows
        self.disposed = False

    def connect(self) -> _FakeConn:
        return _FakeConn(self._rows)

    def dispose(self) -> None:
        self.disposed = True


def _prov(chunk_id: int, source_id: str = "ons-was-wealth", section: str | None = "sec") -> ChunkProvenance:
    return ChunkProvenance(
        chunk_id=chunk_id,
        source_id=source_id,
        document_id=f"doc-{chunk_id}",
        section=section,
        page=None,
        span=f"span-{chunk_id}",
        access_date=_ACCESS_DATE,
    )


def _answer(cited: list[int], fabricated: list[int] | None = None) -> ComposedAnswer:
    return ComposedAnswer(text="...", cited_chunk_ids=cited, fabricated_chunk_ids=fabricated or [])


# ── pure resolution (_resolve_citations) ──────────────────────────────────────


def test_resolves_cited_id_to_full_citation() -> None:
    """A cited, evidence-grounded id becomes a Citation with DB provenance + registry name/URL."""
    answer = _answer([9118])
    provenance = {
        9118: ChunkProvenance(9118, "ons-was-wealth", "was-decile", "Decile 10", None, "decile=10", _ACCESS_DATE)
    }
    result = _resolve_citations(answer, provenance, _REGISTRY)

    assert result.unresolved_chunk_ids == []
    (citation,) = result.citations
    assert citation == Citation(
        chunk_id=9118,
        source_id="ons-was-wealth",
        source_name="ONS Wealth and Assets Survey (WAS)",
        document_id="was-decile",
        section="Decile 10",
        page=None,
        span="decile=10",
        url="https://ons.example/was",
        access_date=_ACCESS_DATE,
    )


def test_fabricated_id_is_stripped_and_flagged() -> None:
    """The done-when: a fabricated cited id is caught; only resolved citations remain.

    777 is in fabricated_chunk_ids (compose saw it was not in the evidence), so
    it is dropped WITHOUT a provenance lookup, even though it has no row here.
    """
    answer = _answer(cited=[9118, 777], fabricated=[777])
    provenance = {9118: _prov(9118)}
    result = _resolve_citations(answer, provenance, _REGISTRY)

    assert [c.chunk_id for c in result.citations] == [9118]
    assert result.unresolved_chunk_ids == [777]


def test_fabricated_id_is_stripped_even_when_it_names_a_real_chunk() -> None:
    """The core guarantee: fabricated wins over a real lookup.

    An id compose flagged as fabricated (not in the evidence the model saw) is
    stripped even if it happens to name a real chunk with a valid registry
    source — provenance the model never saw cannot vouch for a claim.
    """
    answer = _answer(cited=[9118, 777], fabricated=[777])
    provenance = {9118: _prov(9118), 777: _prov(777)}  # 777 IS a real chunk here
    result = _resolve_citations(answer, provenance, _REGISTRY)

    assert [c.chunk_id for c in result.citations] == [9118]
    assert result.unresolved_chunk_ids == [777]


def test_missing_from_corpus_is_stripped() -> None:
    """An evidence-grounded id with no chunk row (data anomaly) is stripped, not served."""
    answer = _answer(cited=[9118, 9999])  # 9999 not fabricated, but absent from the corpus
    provenance = {9118: _prov(9118)}
    result = _resolve_citations(answer, provenance, _REGISTRY)

    assert [c.chunk_id for c in result.citations] == [9118]
    assert result.unresolved_chunk_ids == [9999]


def test_unknown_source_is_stripped() -> None:
    """A real chunk whose source is not in the registry cannot be cited (no name/URL)."""
    answer = _answer(cited=[9118])
    provenance = {9118: _prov(9118, source_id="mystery-source")}
    result = _resolve_citations(answer, provenance, _REGISTRY)

    assert result.citations == []
    assert result.unresolved_chunk_ids == [9118]


def test_order_is_first_appearance_for_resolved_and_unresolved() -> None:
    """Both lists follow the answer's cited order, interleaving stripped ids out."""
    answer = _answer(cited=[9140, 777, 9118], fabricated=[777])
    provenance = {9118: _prov(9118), 9140: _prov(9140, source_id="hmrc-cgt-statistics")}
    result = _resolve_citations(answer, provenance, _REGISTRY)

    assert [c.chunk_id for c in result.citations] == [9140, 9118]
    assert [c.source_name for c in result.citations] == [
        "HMRC Capital Gains Tax Statistics",
        "ONS Wealth and Assets Survey (WAS)",
    ]
    assert result.unresolved_chunk_ids == [777]


def test_no_citations_resolves_to_empty() -> None:
    """A refusal (no cited ids) resolves to an empty result."""
    assert _resolve_citations(_answer([]), {}, _REGISTRY) == ResolvedCitations()


def test_fabricated_id_is_logged_loudly(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="wealthlens_analyst.answer.citations"):
        _resolve_citations(_answer(cited=[777], fabricated=[777]), {}, _REGISTRY)
    assert any("777" in record.getMessage() and "fabricated" in record.getMessage() for record in caplog.records)


# ── row -> provenance mapper (_provenance_from_rows) ───────────────────────────


def test_provenance_from_rows_maps_and_coerces_chunk_id() -> None:
    rows = [
        SimpleNamespace(
            chunk_id=7,
            source_id="ons-was-wealth",
            document_id="d",
            section="s",
            page=3,
            span="sp",
            access_date=_ACCESS_DATE,
        ),
        SimpleNamespace(
            chunk_id=True,
            source_id="hmrc-cgt-statistics",
            document_id="d2",
            section=None,
            page=None,
            span=None,
            access_date=_ACCESS_DATE,
        ),
    ]
    provenance = _provenance_from_rows(rows)

    assert set(provenance) == {7, 1}  # True coerces to int 1
    assert isinstance(provenance[1].chunk_id, int)
    assert provenance[7] == ChunkProvenance(7, "ons-was-wealth", "d", "s", 3, "sp", _ACCESS_DATE)
    assert provenance[1].section is None
    assert provenance[1].span is None


def test_provenance_from_rows_empty() -> None:
    assert _provenance_from_rows([]) == {}


# ── registry loader (load_source_registry) ────────────────────────────────────


def test_load_source_registry_maps_corpus_id_to_name_and_url(tmp_path: Path) -> None:
    registry_file = tmp_path / "sources.yml"
    registry_file.write_text(
        "sources:\n"
        "  - id: a\n    name: A source\n    url: https://a\n    analyst_corpus: true\n"
        "  - id: b\n    name: B source\n    url: https://b\n    analyst_corpus: true\n",
        encoding="utf-8",
    )
    assert load_source_registry(registry_file) == {
        "a": SourceMeta("A source", "https://a"),
        "b": SourceMeta("B source", "https://b"),
    }


def test_load_source_registry_ignores_non_corpus_sources(tmp_path: Path) -> None:
    """Only analyst_corpus sources are loaded; a malformed NON-corpus entry does not break the load.

    Decouples the analyst from edits to the many unrelated (dashboard) sources in
    the shared registry — including a dashboard-only entry left mid-edit with no url.
    """
    registry_file = tmp_path / "sources.yml"
    registry_file.write_text(
        "sources:\n"
        "  - id: corpus\n    name: Corpus source\n    url: https://c\n    analyst_corpus: true\n"
        "  - id: dashboard-only\n    name: Dashboard source\n"  # no url, no analyst_corpus
        "  - id: other-dashboard\n    name: Other\n    url: https://o\n",  # well-formed, non-corpus
        encoding="utf-8",
    )
    assert load_source_registry(registry_file) == {"corpus": SourceMeta("Corpus source", "https://c")}


def test_load_source_registry_fails_on_corpus_entry_missing_url(tmp_path: Path) -> None:
    registry_file = tmp_path / "sources.yml"
    registry_file.write_text("sources:\n  - id: a\n    name: A source\n    analyst_corpus: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing id/name/url"):
        load_source_registry(registry_file)


def test_load_source_registry_fails_on_duplicate_corpus_id(tmp_path: Path) -> None:
    """A repeated corpus id would silently clobber provenance — fail loud instead."""
    registry_file = tmp_path / "sources.yml"
    registry_file.write_text(
        "sources:\n"
        "  - id: dup\n    name: First\n    url: https://first\n    analyst_corpus: true\n"
        "  - id: dup\n    name: Second\n    url: https://second\n    analyst_corpus: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate analyst_corpus source id"):
        load_source_registry(registry_file)


def test_load_source_registry_fails_without_sources_list(tmp_path: Path) -> None:
    registry_file = tmp_path / "sources.yml"
    registry_file.write_text("not_sources: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no top-level 'sources' list"):
        load_source_registry(registry_file)


#: The frozen v1 analyst corpus — every source id tagged analyst_corpus: true in
#: registries/sources.yml. A rename/removal/duplicate of any breaks live citation
#: resolution (resolve_citations strips an uncatalogued source_id as unknown-source).
#: The corpus is FROZEN until v1 ships, so pin the EXACT set: changing it is a
#: deliberate act that must update this test.
_FROZEN_CORPUS_SOURCE_IDS = {
    "ons-was-wealth",
    "hmrc-cgt-statistics",
    "hmrc-tax-receipts",
    "rf-intergenerational-audit-2024",
    "ifs-r188-inheritances-lifecycle",
    "rf-before-the-fall-2025",
    "ifs-green-budget-2023-iht",
    "ifs-deaton-trends-wealth",
}


def test_default_registry_resolves_every_frozen_corpus_source() -> None:
    """The real registries/sources.yml carries exactly the frozen-corpus sources, each citable.

    Reads the committed file (DB-free). Exact-set: catches a corpus source lost to
    a rename/removal AND an unexpected corpus addition; each must resolve to a name
    and an http URL so live citations render.
    """
    registry = load_source_registry()
    assert set(registry) == _FROZEN_CORPUS_SOURCE_IDS
    for source_id in _FROZEN_CORPUS_SOURCE_IDS:
        assert registry[source_id].name
        assert registry[source_id].url.startswith("http")


# ── resolve_citations short-circuits (no DB when no lookup is needed) ──────────


def test_resolve_citations_uncited_answer_opens_no_connection() -> None:
    result = resolve_citations(_answer([]), engine=_ExplodingEngine(), registry=_REGISTRY)  # type: ignore[arg-type]
    assert result == ResolvedCitations()


def test_resolve_citations_all_fabricated_opens_no_connection() -> None:
    answer = _answer(cited=[777, 888], fabricated=[777, 888])
    result = resolve_citations(answer, engine=_ExplodingEngine(), registry=_REGISTRY)  # type: ignore[arg-type]
    assert result.citations == []
    assert result.unresolved_chunk_ids == [777, 888]


def test_resolve_citations_defaults_registry_from_disk_without_db() -> None:
    """registry=None loads the real file; an all-fabricated answer still needs no DB."""
    answer = _answer(cited=[777], fabricated=[777])
    result = resolve_citations(answer, engine=_ExplodingEngine())  # type: ignore[arg-type]
    assert result.unresolved_chunk_ids == [777]


def test_resolve_citations_fetches_rows_and_does_not_dispose_a_supplied_engine() -> None:
    """The DB-fetch path resolves rows to citations and leaves a caller-supplied engine alone.

    A caller-supplied (shared app) engine is the caller's to manage — disposing it
    would tear down a pool still in use in the H1-20 request path.
    """
    answer = _answer(cited=[9118])
    rows = [
        SimpleNamespace(
            chunk_id=9118,
            source_id="ons-was-wealth",
            document_id="was-decile",
            section="Decile 10",
            page=None,
            span="decile=10",
            access_date=_ACCESS_DATE,
        )
    ]
    engine = _FakeEngine(rows)
    result = resolve_citations(answer, engine=engine, registry=_REGISTRY)  # type: ignore[arg-type]

    assert [c.chunk_id for c in result.citations] == [9118]
    assert result.citations[0].source_name == "ONS Wealth and Assets Survey (WAS)"
    assert result.citations[0].url == "https://ons.example/was"
    assert result.citations[0].span == "decile=10"
    assert result.citations[0].access_date == _ACCESS_DATE
    assert engine.disposed is False


def test_resolve_citations_disposes_a_self_built_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """When it builds its own engine (no engine injected), resolve_citations disposes it."""
    answer = _answer(cited=[9118])
    rows = [
        SimpleNamespace(
            chunk_id=9118,
            source_id="ons-was-wealth",
            document_id="d",
            section="s",
            page=None,
            span="sp",
            access_date=_ACCESS_DATE,
        )
    ]
    engine = _FakeEngine(rows)
    monkeypatch.setattr(citations, "load_settings", lambda: object())
    monkeypatch.setattr(citations, "engine_from_settings", lambda settings: engine)
    result = resolve_citations(answer, registry=_REGISTRY)

    assert [c.chunk_id for c in result.citations] == [9118]
    assert engine.disposed is True


def test_default_registry_path_honors_repo_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """WEALTHLENS_REPO_ROOT overrides discovery for non-editable/deployed installs (mirrors ingest)."""
    from wealthlens_analyst.answer.citations import _default_registry_path

    monkeypatch.setenv("WEALTHLENS_REPO_ROOT", "/opt/wealthlens")
    assert _default_registry_path() == Path("/opt/wealthlens") / "registries" / "sources.yml"
