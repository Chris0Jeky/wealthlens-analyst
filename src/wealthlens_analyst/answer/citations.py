"""Resolve and verify chunk-level citations (ADR 0001 §4).

A cited chunk_id is only served if it resolves to a real chunk in the corpus
AND that chunk was part of the evidence the model was actually shown.
Resolution attaches the ingestion-time provenance (source_id, document_id,
section, page) and the source's citable name + URL from the registry
(registries/sources.yml). Anything that does not resolve is stripped and
flagged: a fabricated citation is worse than no answer, so this module never
lets one reach the user.

Three unresolvable cases, all stripped and reported (loudly logged, and
returned in ``ResolvedCitations.unresolved_chunk_ids``):

- **fabricated** — the id was not in the supplied evidence. compose.py already
  marks these in ``ComposedAnswer.fabricated_chunk_ids``; provenance the model
  never saw cannot vouch for a claim, even in the rare case that the id happens
  to name a real chunk, so it is dropped without a lookup.
- **missing** — the id has no row in the ``chunks`` table. For an
  evidence-grounded id this is a data anomaly (e.g. the chunk was re-ingested
  away mid-request); for anything else it is a plain fabrication.
- **unknown-source** — the chunk exists but its ``source_id`` is absent from the
  registry, so no citable source name/URL can be produced. That is a
  provenance/registry integrity fault (ingestion admitted a chunk whose source
  is not catalogued), and a citation with no source is not a citation.

Citation resolvability is one of the deterministic eval checks (H1-20/H1-23),
so the resolution logic is pure and DB-free-testable: the database fetch and
the registry load are thin wrappers around ``_resolve_citations``, which the
done-when fixture exercises without a database (CI has no Postgres).

Task H1-19 in tasks/hero1-backlog.md.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import yaml
from sqlalchemy import Engine

from wealthlens_analyst.answer.compose import ComposedAnswer
from wealthlens_analyst.config import load_settings
from wealthlens_analyst.db import engine_from_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceMeta:
    """A source's citable identity, from registries/sources.yml."""

    name: str
    url: str


@dataclass(frozen=True)
class Citation:
    """A resolved, render-ready citation.

    Carries the full ADR 0001 §4 provenance tuple (section, page, span) plus the
    source's registry name + URL and the ingestion ``access_date`` — the
    data-integrity rule requires every citation to carry both a URL and an
    access date, and ``span`` disambiguates repeated section labels / coarse
    pages, so nothing that the DB stores about a citation's origin is dropped
    here.
    """

    chunk_id: int
    source_id: str
    source_name: str
    document_id: str
    section: str | None
    page: int | None
    span: str | None
    url: str  # the registry source URL (registries/sources.yml)
    access_date: date  # when the source was fetched (chunks.access_date, NOT NULL)


@dataclass(frozen=True)
class ChunkProvenance:
    """Ingestion-time provenance for one chunk, as stored in the chunks table.

    The citation-relevant subset of the chunks row (ADR 0001 §4): identity, where
    in the source it came from (section/page/span), and the access date. No
    text — resolution is about provenance, not re-fetching the evidence body.
    """

    chunk_id: int
    source_id: str
    document_id: str
    section: str | None
    page: int | None
    span: str | None
    access_date: date


@dataclass(frozen=True)
class ResolvedCitations:
    """The outcome of resolving an answer's cited ids against the corpus.

    ``citations`` are render-ready and ordered by first appearance in the
    answer. ``unresolved_chunk_ids`` are the cited ids that were stripped
    (fabricated, missing from the corpus, or from an unknown source), in the
    same order — machine-readable so the /ask response (H1-20) can flag that an
    answer's citations were pruned rather than silently serve fewer than the
    model claimed.
    """

    citations: list[Citation] = field(default_factory=list)
    unresolved_chunk_ids: list[int] = field(default_factory=list)


# Fetch provenance for a set of cited ids. The IN-list binds via an EXPANDING
# bindparam (SQLAlchemy renders one placeholder per id, safely); the caller
# short-circuits an empty id list so `IN ()` is never generated. Selects the
# ADR 0001 provenance columns a Citation needs (not the chunk text or ts).
_RESOLVE_SQL = sa.text(
    """
    SELECT chunk_id, source_id, document_id, section, page, span, access_date
    FROM chunks
    WHERE chunk_id IN :chunk_ids
    """
).bindparams(sa.bindparam("chunk_ids", expanding=True))


def _provenance_from_rows(rows: Iterable[Any]) -> dict[int, ChunkProvenance]:
    """Map chunk rows to a ``{chunk_id: ChunkProvenance}`` lookup.

    Pure and DB-free (rows carry attribute access like a SQLAlchemy Row, but
    could equally be test stand-ins), so the provenance mapping is unit-tested
    without a database — the SQL itself is verified live. chunk_id is coerced
    to ``int`` so the lookup key type matches the parsed cited ids.
    """
    provenance: dict[int, ChunkProvenance] = {}
    for row in rows:
        chunk_id = int(row.chunk_id)
        provenance[chunk_id] = ChunkProvenance(
            chunk_id=chunk_id,
            source_id=row.source_id,
            document_id=row.document_id,
            section=row.section,
            page=row.page,
            span=row.span,
            access_date=row.access_date,
        )
    return provenance


def _resolve_citations(
    answer: ComposedAnswer,
    provenance_by_id: Mapping[int, ChunkProvenance],
    registry: Mapping[str, SourceMeta],
) -> ResolvedCitations:
    """Resolve cited ids to citations; strip and flag the unresolvable ones.

    Pure so the done-when (a fabricated id is caught; only resolved citations
    remain) is exercised without a database. Output order follows the answer's
    first-appearance cited order, for both resolved citations and the unresolved
    id list.
    """
    fabricated = set(answer.fabricated_chunk_ids)
    citations: list[Citation] = []
    unresolved: list[int] = []
    for chunk_id in answer.cited_chunk_ids:
        if chunk_id in fabricated:
            # Not in the evidence the model was shown — compose already flagged
            # it. A citation to provenance the model never saw is never served.
            logger.warning("citations: dropping fabricated cited id %d (not in the supplied evidence)", chunk_id)
            unresolved.append(chunk_id)
            continue
        provenance = provenance_by_id.get(chunk_id)
        if provenance is None:
            # Cited and evidence-grounded per compose, yet absent from the
            # corpus: a data anomaly (e.g. re-ingested away mid-request).
            logger.warning("citations: dropping cited id %d (no chunk row in the corpus)", chunk_id)
            unresolved.append(chunk_id)
            continue
        source = registry.get(provenance.source_id)
        if source is None:
            # The chunk exists but its source is not catalogued, so there is no
            # citable name/URL — a provenance/registry integrity fault.
            logger.warning(
                "citations: dropping cited id %d (source_id %r has no registry entry)",
                chunk_id,
                provenance.source_id,
            )
            unresolved.append(chunk_id)
            continue
        citations.append(
            Citation(
                chunk_id=chunk_id,
                source_id=provenance.source_id,
                source_name=source.name,
                document_id=provenance.document_id,
                section=provenance.section,
                page=provenance.page,
                span=provenance.span,
                url=source.url,
                access_date=provenance.access_date,
            )
        )
    return ResolvedCitations(citations=citations, unresolved_chunk_ids=unresolved)


def _default_registry_path() -> Path:
    """Locate ``registries/sources.yml`` at the repo root.

    ``WEALTHLENS_REPO_ROOT`` overrides the search, for a non-editable install or
    deployed service where the package no longer sits inside the checked-out
    tree (the H1-30 shape). Otherwise walk up from this file to the ancestor
    holding both ``registries/`` and ``projects/``. Mirrors
    ingest.slice_corpus._find_repo_root exactly, so both the ingest and the
    citation paths resolve the shared registry the same way.
    """
    override = os.environ.get("WEALTHLENS_REPO_ROOT")
    if override:
        return Path(override) / "registries" / "sources.yml"
    for parent in Path(__file__).resolve().parents:
        if (parent / "registries").is_dir() and (parent / "projects").is_dir():
            return parent / "registries" / "sources.yml"
    raise RuntimeError(
        "could not locate registries/sources.yml (no ancestor has both registries/ and projects/); "
        "set WEALTHLENS_REPO_ROOT to override"
    )


def load_source_registry(path: Path | None = None) -> dict[str, SourceMeta]:
    """Load ``{source_id: SourceMeta}`` for the citable (analyst-corpus) sources.

    Only entries tagged ``analyst_corpus: true`` are loaded. A citation's
    ``source_id`` is always a frozen-corpus source (the corpus is frozen until
    v1 ships), so the many unrelated dashboard/pipeline sources in this shared
    repo-root registry are irrelevant here — and skipping them keeps the
    analyst's citation resolution decoupled from edits to sources it never
    cites (a malformed dashboard-only entry must not break the analyst).

    Fails loudly on a structural fault or a malformed/duplicated CORPUS entry:
    a missing id/name/url, or a repeated id (which would otherwise silently
    last-wins and attach the WRONG source name/URL to every citation for that
    source). A citation needs correct provenance, so a broken corpus entry is a
    hard error, not something to paper over — consistent with every other
    integrity guard here. The request path (H1-20) loads this once at startup
    and injects it; the standalone/CLI path lets it default.
    """
    registry_path = path or _default_registry_path()
    with registry_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{registry_path} is not a YAML mapping")
    entries = data.get("sources")
    if not isinstance(entries, list):
        raise ValueError(f"{registry_path} has no top-level 'sources' list")
    registry: dict[str, SourceMeta] = {}
    for entry in entries:
        # Ignore anything that is not a citable corpus source — the analyst
        # neither cites nor validates unrelated (e.g. dashboard-only) sources,
        # so their shape is not its concern.
        if not isinstance(entry, dict) or not entry.get("analyst_corpus"):
            continue
        source_id = entry.get("id")
        name = entry.get("name")
        url = entry.get("url")
        if not source_id or not name or not url:
            raise ValueError(f"{registry_path}: analyst_corpus source entry missing id/name/url: {entry!r}")
        key = str(source_id)
        if key in registry:
            raise ValueError(f"{registry_path}: duplicate analyst_corpus source id {source_id!r}")
        registry[key] = SourceMeta(name=str(name), url=str(url))
    return registry


def resolve_citations(
    answer: ComposedAnswer,
    *,
    engine: Engine | None = None,
    registry: Mapping[str, SourceMeta] | None = None,
) -> ResolvedCitations:
    """Resolve an answer's cited chunk_ids to render-ready citations.

    Fetches provenance for the cited, evidence-grounded ids from the ``chunks``
    table and attaches each source's registry name + URL; fabricated, missing,
    and unknown-source ids are stripped and returned in
    ``ResolvedCitations.unresolved_chunk_ids`` (all logged).

    ``engine`` and ``registry`` are injected by the request path (H1-20: the
    shared app engine plus a registry loaded once at startup). When omitted they
    default to an environment-built engine — disposed after use, since it owns
    its pool, mirroring search_fts/search_dense — and a freshly loaded registry,
    for standalone/CLI/live-check use. Only ids that actually need a lookup hit
    the database; an answer that cites nothing (a refusal) never opens a
    connection.
    """
    if not answer.cited_chunk_ids:
        # A refusal sentence (or any uncited answer) resolves to nothing without
        # touching the database or the registry.
        return ResolvedCitations()
    if registry is None:
        registry = load_source_registry()
    fabricated = set(answer.fabricated_chunk_ids)
    # Fabricated ids are unresolvable by definition, so they need no lookup;
    # only evidence-grounded ids are fetched. cited_chunk_ids is already
    # deduplicated (compose), so this preserves order and uniqueness.
    to_fetch = [chunk_id for chunk_id in answer.cited_chunk_ids if chunk_id not in fabricated]
    if not to_fetch:
        # Every cited id is fabricated: all unresolved, no DB work.
        return _resolve_citations(answer, {}, registry)
    # Only dispose an engine WE created — a caller-supplied (shared) engine is
    # the caller's to manage (mirrors retrieval/fts.py::search_fts).
    created = engine is None
    if engine is None:
        engine = engine_from_settings(load_settings())
    try:
        with engine.connect() as conn:
            rows = conn.execute(_RESOLVE_SQL, {"chunk_ids": to_fetch}).all()
    finally:
        if created:
            engine.dispose()
    return _resolve_citations(answer, _provenance_from_rows(rows), registry)
