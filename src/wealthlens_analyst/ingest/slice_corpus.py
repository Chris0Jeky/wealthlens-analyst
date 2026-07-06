"""Ingest the frozen corpus slice into the chunks table.

Two ingestion paths, one provenance contract (ADR 0001 §4):

1. Tabular sources (ONS WAS, HMRC) — rendered to citable text from the
   EXISTING pipelines' processed outputs (automation/data-pipelines/ writes
   projects/wealthlens-dashboard/data/processed/*.csv). One chunk per table
   section/band, carrying period and units, so a citation can say exactly which
   statistic backed a claim. Implemented here (task H1-07).
2. Report documents (IFS/RF PDFs from fetch_documents.py) — page-aware
   extraction, ~500-token heading-anchored chunks with page + span recorded.
   Pending task H1-08.

Every chunk row carries source_id (registries/sources.yml id), document_id,
section, page, span. The per-type rule (tabular: section + span, no page;
document: page + span) is enforced at write time (task H1-09); the renderers
here always emit conformant chunks.

Data-honesty guardrails baked into the tabular renderer:
- A missing/suppressed cell is OMITTED, never rendered as a fabricated 0. A
  genuine zero in the source still renders as "0", and a sub-microunit non-zero
  keeps its value rather than collapsing to "0".
- The honesty filter is an ALLOWLIST (fail-closed): a row is ingested only when
  its data_source marker is a known-official value (e.g. "live"). Anything else
  — an illustrative fallback, an unrecognised marker, a blank, or a missing
  column — is SKIPPED, so the analyst never cites fabricated or unverified
  figures as official facts. (This mirrors the spend cap's fail-closed posture.)

Size-threshold tables (HMRC CGT) get two extra, source-faithful renderings so a
citation can state concentration directly (seeded follow-up from the H1-07
review; serves golden G-007/G-008):
- each band's label becomes an explicit RANGE — its lower bound "to under" the
  next band's lower bound (the bands are half-open), the top band reading "and
  above" — so HMRC's "£X+" band name is never misread as an "above £X"
  cumulative threshold; and
- a CUMULATIVE concentration clause from the cumul_*_from_top columns, e.g.
  "taxpayers with gains of £X and above accounted for Y% of all taxable gains
  and made up Z% of all CGT taxpayers" — each share carries its own verb so a
  money share and a headcount share are never grammatically conflated. Shares
  are clamped to <=100% to absorb the rounding artefact of summing
  already-rounded per-band shares (a value beyond tolerance is OMITTED, not
  shown wrong — the same fail-closed posture).
The span still pins the raw HMRC band label, so provenance stays faithful while
the rendered text is unambiguous.

Entrypoint: `make ingest-slice` -> main() (task H1-09): collect chunks ->
validate provenance (fail-closed) -> write to the chunks table. The STORED
tsvector (migration 0001_chunks) keeps the FTS index current on every insert,
so ingestion cannot forget it; dense embeddings are a separate step (task
H1-11). The PDF document path (task H1-08) is not built yet, so today's slice is
the tabular sources only.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import Engine

from wealthlens_analyst.config import load_settings
from wealthlens_analyst.db import chunks_table, engine_from_settings
from wealthlens_analyst.retrieval.dense import embed_corpus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A corpus chunk with ingestion-time provenance (mirrors the chunks table).

    The provenance fields map 1:1 onto the chunks columns (migration
    0001_chunks). For tabular chunks ``section`` and ``span`` are always set and
    ``page`` is always None; the write path (H1-09) enforces that per-type rule.

    ``token_count`` is an approximate whitespace word count. Tabular chunks are
    short, so an exact tokenizer is unnecessary here; the document path (H1-08)
    that targets ~500-token chunks introduces a real tokenizer for that column.
    """

    source_id: str  # registries/sources.yml id — the citation root
    document_id: str  # stable id of the table/document within the source
    section: str | None
    page: int | None
    span: str | None
    text: str
    token_count: int
    access_date: date


@dataclass(frozen=True)
class ValueColumn:
    """One statistic column to render, with its human label and units.

    ``prefix``/``suffix`` carry their own spacing so the rendered figure reads
    naturally: a currency column uses prefix="£", suffix="bn" ("£13.9bn"); a
    percentage uses suffix="%" ("0.7%"); a word unit uses a leading-space suffix
    (" thousand" -> "61 thousand").
    """

    column: str
    label: str
    prefix: str = ""
    suffix: str = ""


@dataclass(frozen=True)
class CumulativeColumn:
    """One cumulative-from-the-top share folded into a size-band's chunk text.

    For a size-threshold table (``TableSpec.band_lower_column`` set) this names a
    column holding a cumulative-from-the-top share. ``verb`` is the predicate that
    correctly attaches the share to the clause subject — the population at the
    threshold and above — so a money share and a headcount share are never
    grammatically conflated (e.g. "accounted for" for the gains share, "made up"
    for the taxpayer-count share). ``of_label`` is the denominator phrase
    ("all taxable gains"); ``suffix`` is the unit ("%").
    """

    column: str
    verb: str
    of_label: str
    suffix: str = "%"


@dataclass(frozen=True)
class CumulativeSpec:
    """Render a cumulative-concentration clause for a size-threshold band table.

    Valid only when ``TableSpec.band_lower_column`` is set (the bands are size
    thresholds). ``subject`` names the population at the threshold and above
    ("taxpayers with gains of" -> "taxpayers with gains of £X and above"); each
    column then states what that population did via its own ``verb``, so a money
    share and a headcount share keep distinct, correct grammar. Each cumulative
    share is clamped to ``<= max_pct`` to absorb the rounding artefact of
    cumulative-summing already-rounded per-band shares; a value beyond
    ``max_pct + tolerance`` is OMITTED (fail-closed) rather than shown wrong.
    """

    subject: str
    columns: tuple[CumulativeColumn, ...]
    max_pct: float = 100.0
    tolerance: float = 1.0


@dataclass(frozen=True)
class TableSpec:
    """How to render one processed tabular source into citable chunks."""

    source_id: str
    csv_name: str  # processed CSV under projects/wealthlens-dashboard/data/processed/
    document_id: str
    table_label: str  # human table name, used in the chunk text and section
    period: str  # the period/edition the table covers, surfaced in every citation
    section_column: str  # the column whose value names each band/section
    section_noun: str  # what a band is, e.g. "wealth decile", "size-of-gain band"
    value_columns: tuple[ValueColumn, ...]
    # access_date mirrors registries/sources.yml; H1-09 will load it from the
    # registry directly (the analyst has no YAML dep yet — see ADR 0001).
    access_date: date
    # Honesty filter (ALLOWLIST, fail-closed). If set, a row is ingested only
    # when row[honesty_flag_column].strip().lower() is in official_markers;
    # everything else (illustrative/unknown/blank/missing) is skipped. None means
    # the table has no honesty flag (it is wholly official, e.g. ONS WAS / HMRC).
    honesty_flag_column: str | None = None
    official_markers: tuple[str, ...] = ("live", "official")
    # Size-threshold tables (HMRC CGT) only; both default OFF so wholly
    # categorical tables (ONS WAS deciles, HMRC receipts by year) are unaffected.
    # When band_lower_column is set, each band's displayed label becomes an
    # explicit numeric range (its lower bound to the next band's lower bound; the
    # top band reads "and above"), formatted with band_range_prefix. When
    # cumulative is set, a cumulative-from-the-top concentration clause is
    # appended to each band's chunk.
    band_lower_column: str | None = None
    band_range_prefix: str = "£"
    cumulative: CumulativeSpec | None = None

    def __post_init__(self) -> None:
        # The cumulative clause is anchored on each band's lower bound, so it
        # needs the lower-bound column to know the "£X and above" threshold.
        if self.cumulative is not None and self.band_lower_column is None:
            raise ValueError(f"TableSpec {self.source_id!r}: cumulative rendering requires band_lower_column")


# The frozen tabular slice: the analyst_corpus tabular sources in
# registries/sources.yml, each mapped to its existing processed pipeline output.
TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        source_id="ons-was-wealth",
        csv_name="ons_wealth_by_decile.csv",
        document_id="was-total-wealth-by-decile",
        table_label="Total household wealth by decile, Great Britain",
        period="ONS Wealth and Assets Survey, April 2020 to March 2022",
        section_column="decile",
        section_noun="wealth decile",
        value_columns=(ValueColumn("total_wealth_bn", "aggregate total net wealth", prefix="£", suffix="bn"),),
        access_date=date(2026, 5, 30),  # registries/sources.yml: ons-was-wealth
    ),
    TableSpec(
        source_id="hmrc-cgt-statistics",
        csv_name="hmrc_cgt_concentration.csv",
        document_id="hmrc-cgt-size-of-gain",
        table_label="Capital Gains Tax by size of gain",
        period="HMRC Capital Gains Tax statistics, 2025 edition",
        section_column="gain_band",
        section_noun="size-of-gain band",
        # Per-band columns are INCREMENTAL for the band [band_lower[i],
        # band_lower[i+1]) (fetch_hmrc_stats.process: share = band gains / total),
        # so they are labelled "in this band". The displayed band label is an
        # explicit RANGE (band_lower_column below) so HMRC's "£X+" band name is
        # not misread as "above £X". The cumulative top-tail concentration is
        # rendered separately, from the cumul_*_from_top columns (see cumulative=).
        value_columns=(
            ValueColumn("num_taxpayers_thousands", "taxpayers with gains in this band", suffix=" thousand"),
            ValueColumn("total_gains_millions", "total gains in this band", prefix="£", suffix="m"),
            ValueColumn("share_of_gains_pct", "share of all taxable gains in this band", suffix="%"),
            ValueColumn("share_of_taxpayers_pct", "share of all CGT taxpayers in this band", suffix="%"),
        ),
        access_date=date(2026, 5, 16),  # registries/sources.yml: hmrc-cgt-statistics
        # Explicit band ranges + a cumulative-from-the-top concentration clause.
        # cumul_*_from_top_pct are cumulative sums of the 1-dp per-band shares, so
        # the low bands round to ~100.1/100.6%; the CumulativeSpec clamp absorbs
        # that (and omits anything beyond tolerance). Directly serves golden
        # G-007 (concentration) and G-008 (share of gains above £1m).
        band_lower_column="band_lower",
        cumulative=CumulativeSpec(
            # Subject is the PEOPLE at the threshold and above; each column gets a
            # verb that fits them, so "...made up 2.8% of all CGT taxpayers" is not
            # rendered as the category error "gains ... account for ... taxpayers".
            subject="taxpayers with gains of",
            columns=(
                CumulativeColumn("cumul_gains_from_top_pct", "accounted for", "all taxable gains"),
                CumulativeColumn("cumul_taxpayers_from_top_pct", "made up", "all CGT taxpayers"),
            ),
        ),
    ),
    TableSpec(
        source_id="hmrc-tax-receipts",
        csv_name="tax_composition.csv",
        document_id="hmrc-tax-nic-receipts",
        table_label="UK tax and NIC receipts by tax",
        period="HMRC Tax and NIC Receipts, April 2025 edition",
        section_column="year",
        section_noun="financial year",
        # Only the raw HMRC receipt line items are ingested. The work-vs-wealth
        # split (work_pct/wealth_pct) is OUR editorial derivation, not an HMRC
        # figure, so it is deliberately not cited under the HMRC attribution; it
        # can be derived at answer time.
        value_columns=(
            ValueColumn("income_tax_bn", "Income Tax receipts", prefix="£", suffix="bn"),
            ValueColumn("nics_bn", "National Insurance contributions", prefix="£", suffix="bn"),
            ValueColumn("cgt_bn", "Capital Gains Tax receipts", prefix="£", suffix="bn"),
            ValueColumn("iht_bn", "Inheritance Tax receipts", prefix="£", suffix="bn"),
            ValueColumn("sdlt_bn", "Stamp Duty Land Tax receipts", prefix="£", suffix="bn"),
        ),
        access_date=date(2026, 5, 16),  # registries/sources.yml: hmrc-tax-receipts
        # The dashboard pipeline emits data_source="live" for real receipts and
        # "illustrative"/"unknown" for the fallback; only "live" is ingested.
        honesty_flag_column="data_source",
    ),
)


def _parse_float(raw: str | None) -> float | None:
    """Parse one CSV cell into a finite float, or None if it is not usable.

    Returns None for blank/missing/non-numeric/non-finite cells (so a suppressed
    cell is never coerced into a number). Thousands separators in the input are
    tolerated.
    """
    if raw is None:
        return None
    text = raw.strip().replace(",", "")  # tolerate thousands separators in the source
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _format_number(value: float) -> str:
    """Format a finite float for display: noise rounded, thousands grouped.

    A genuine "0" stays "0"; a sub-microunit non-zero keeps significant figures
    rather than collapsing to "0" (which would assert a zero the source did not
    have). Float-repr noise (100.10000000000001) is rounded away.
    """
    rounded = round(value, 6)  # strip float-repr noise
    if rounded == 0 and value != 0:
        return f"{value:g}"
    if rounded == int(rounded):
        return f"{int(rounded):,}"
    return f"{rounded:,}"


def _clean_number(raw: str | None) -> str | None:
    """Parse one CSV cell into a display number, or None if it is not usable.

    None for blank/missing/non-numeric/non-finite cells so the caller OMITS them
    — a suppressed cell must never become a fabricated 0.
    """
    value = _parse_float(raw)
    return None if value is None else _format_number(value)


def _clean_cumulative_pct(raw: str | None, max_pct: float, tolerance: float) -> str | None:
    """Format a cumulative-share cell, clamping the rounding overflow to max_pct.

    A cumulative share of a whole cannot exceed ``max_pct`` (100%), but summing
    already-rounded per-band shares overshoots slightly (~100.1/100.6%). A value
    in ``(max_pct, max_pct + tolerance]`` is clamped to ``max_pct``; a value
    beyond that is OMITTED (returns None) rather than shown wrong — the same
    fail-closed posture as the honesty allowlist. A cumulative share also cannot
    be NEGATIVE (it is a sum of non-negative per-band shares), so a negative value
    is a corruption/anomaly and is likewise OMITTED. Blank/missing/non-finite -> None.
    """
    value = _parse_float(raw)
    if value is None or value < 0.0:
        return None
    if value > max_pct:
        if value > max_pct + tolerance:
            return None  # beyond rounding tolerance -> a real anomaly, omit
        value = max_pct  # absorb the cumulative-rounding overflow
    return _format_number(value)


def _money(value: float, prefix: str) -> str:
    """Format a whole-pound bound as a grouped currency string ("£1,000,000").

    HMRC band bounds are whole pounds; round() (no ndigits) returns an int, which
    formats with a thousands separator.
    """
    return f"{prefix}{round(value):,}"


def _range_label(lower: float, upper: float | None, prefix: str) -> str:
    """Render a band as an explicit range, or "£X and above" for the top band.

    The bands are half-open [lower, upper) (the per-band figure covers gains at or
    above ``lower`` and below the next band's lower bound), so the closed end is
    spelled "to under £Y" — HMRC's own band-table convention — to avoid implying
    £Y is included.
    """
    if upper is None:
        return f"{_money(lower, prefix)} and above"
    return f"{_money(lower, prefix)} to under {_money(upper, prefix)}"


def _join_clauses(parts: list[str]) -> str:
    """Join fragments in British list style: "a", "a and b", "a, b and c"."""
    if len(parts) <= 1:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _statement(column: ValueColumn, value: str) -> str:
    """Render one "<label> <prefix><value><suffix>" statistic fragment."""
    return f"{column.label} {column.prefix}{value}{column.suffix}"


def _capitalise_first(text: str) -> str:
    """Upper-case only the first character (str.capitalize lower-cases the rest,
    which would mangle an acronym section_noun like 'IHT band')."""
    return text[:1].upper() + text[1:]


def _next_bound_map(rows: list[Mapping[str, str]], band_lower_column: str | None) -> dict[float, float]:
    """Map each band's lower bound to the next-higher bound (its upper edge).

    Built from ALL parseable bounds in the table (independent of row order or
    which rows get chunked), so a band's upper edge is the next band's lower
    edge in the source — exactly HMRC's band structure. The highest bound has no
    successor (it is the open-ended "and above" band). Returns {} when the table
    has no bound column.
    """
    if band_lower_column is None:
        return {}
    bounds = sorted({b for row in rows if (b := _parse_float(row.get(band_lower_column))) is not None})
    return {lower: bounds[i + 1] for i, lower in enumerate(bounds) if i + 1 < len(bounds)}


def _cumulative_clause(row: Mapping[str, str], spec: TableSpec, bound: float | None) -> str:
    """Render the cumulative "<subject> £X and above <verb> Y% of …" clause.

    Each column supplies its own verb so the share attaches to the subject with
    correct grammar (the gains share and the taxpayer-count share never share one
    verb). Empty string when the table has no cumulative spec, the band has no
    parseable lower bound, or every cumulative cell is missing/suppressed/
    out-of-tolerance.
    """
    if spec.cumulative is None or bound is None:
        return ""
    parts = [
        f"{cc.verb} {value}{cc.suffix} of {cc.of_label}"
        for cc in spec.cumulative.columns
        if (value := _clean_cumulative_pct(row.get(cc.column), spec.cumulative.max_pct, spec.cumulative.tolerance))
        is not None
    ]
    if not parts:
        return ""
    threshold = _money(bound, spec.band_range_prefix)
    return f" Cumulatively, {spec.cumulative.subject} {threshold} and above {_join_clauses(parts)}."


def render_table_chunks(rows: Iterable[Mapping[str, str]], spec: TableSpec) -> list[Chunk]:
    """Render one processed tabular source into citable chunks (one per band).

    Each chunk's text states the table, the period, the band, and every
    available statistic with its units. Missing cells are omitted (never
    fabricated as 0); rows that are not from a known-official data_source are
    skipped (fail-closed). ``span`` is disambiguated if a band label repeats.

    Size-threshold tables (band_lower_column set) render the band as an explicit
    range and append a cumulative top-tail concentration clause; the span still
    pins the raw band label so provenance stays faithful.
    """
    row_list = list(rows)  # materialised: we look ahead for each band's upper bound
    next_bound = _next_bound_map(row_list, spec.band_lower_column)

    chunks: list[Chunk] = []
    skipped_unofficial = 0
    seen_bands: dict[str, int] = {}
    for row in row_list:
        band = (row.get(spec.section_column) or "").strip()
        if not band:
            continue  # not a data row (blank band label)

        if spec.honesty_flag_column is not None:
            marker = (row.get(spec.honesty_flag_column) or "").strip().lower()
            if marker not in spec.official_markers:
                skipped_unofficial += 1
                continue

        statements = [
            _statement(vc, value)
            for vc in spec.value_columns
            if (value := _clean_number(row.get(vc.column))) is not None
        ]
        if not statements:
            continue  # every value in this row was missing/suppressed

        # The band's lower bound (size-threshold tables only); drives both the
        # explicit range label and the cumulative clause's "£X and above".
        bound = _parse_float(row.get(spec.band_lower_column)) if spec.band_lower_column else None

        # Displayed label: an explicit range when we have a bound, else the raw
        # band label (honest fallback — never a fabricated range).
        display_band = band
        if bound is not None:
            display_band = _range_label(bound, next_bound.get(bound), spec.band_range_prefix)

        cumulative_clause = _cumulative_clause(row, spec, bound)

        # span pins the exact row (the RAW band label, faithful to the source);
        # disambiguate if a band label ever repeats so two chunks never share
        # (source_id, document_id, span).
        occurrence = seen_bands.get(band, 0) + 1
        seen_bands[band] = occurrence
        span = f"{spec.section_column}={band}"
        if occurrence > 1:
            span = f"{span}#{occurrence}"

        text = (
            f"{spec.table_label} ({spec.period}). "
            f"{_capitalise_first(spec.section_noun)} {display_band}: "
            f"{'; '.join(statements)}."
            f"{cumulative_clause}"
        )
        chunks.append(
            Chunk(
                source_id=spec.source_id,
                document_id=spec.document_id,
                section=f"{spec.table_label}: {display_band}",
                page=None,
                span=span,
                text=text,
                token_count=len(text.split()),
                access_date=spec.access_date,
            )
        )

    if skipped_unofficial:
        logger.warning(
            "tabular source %s: skipped %d row(s) whose %s was not in %s (not ingested as citable official facts)",
            spec.source_id,
            skipped_unofficial,
            spec.honesty_flag_column,
            spec.official_markers,
        )
    return chunks


def _find_repo_root() -> Path:
    """Locate the repo root (the dir holding both registries/ and projects/).

    WEALTHLENS_REPO_ROOT overrides the search, for non-editable installs where
    the package no longer sits inside the checked-out tree. Otherwise we walk up
    from this module (reliable under the editable install, same approach as the
    sim registry loaders).
    """
    override = os.environ.get("WEALTHLENS_REPO_ROOT")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "registries").is_dir() and (parent / "projects").is_dir():
            return parent
    raise RuntimeError(
        "could not locate repo root (no ancestor has both registries/ and projects/); "
        "set WEALTHLENS_REPO_ROOT to override"
    )


def collect_tabular_chunks(processed_dir: Path | None = None) -> list[Chunk]:
    """Read every tabular-source CSV and render its chunks.

    ``processed_dir`` defaults to the dashboard pipeline output directory. A CSV
    that is absent (the processed outputs are gitignored build artifacts) is
    skipped with a warning rather than failing — the write path (H1-09) decides
    whether an empty slice is acceptable. Files are read as utf-8-sig so a BOM
    (e.g. from a spreadsheet export) does not corrupt the first column header.
    """
    if processed_dir is None:
        processed_dir = _processed_dir_default()

    chunks: list[Chunk] = []
    for spec in TABLE_SPECS:
        csv_path = processed_dir / spec.csv_name
        if not csv_path.is_file():
            logger.warning(
                "tabular source %s: %s not found, skipping (regenerate via the pipeline)",
                spec.source_id,
                csv_path.name,
            )
            continue
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        rendered = render_table_chunks(rows, spec)
        logger.info("tabular source %s: %d chunk(s) from %d row(s)", spec.source_id, len(rendered), len(rows))
        chunks.extend(rendered)
    return chunks


# --- Ingestion integrity gate + write path (H1-09) -----------------------------


class ProvenanceError(ValueError):
    """A chunk violates the ingestion provenance contract (ADR 0001 §4).

    Raised by validate_chunk_provenance when a chunk lacks the provenance a
    citation needs. Ingestion is fail-closed: write_chunks validates every chunk
    BEFORE writing any row, so one offending chunk aborts the whole batch with
    nothing persisted — the corpus can never hold a chunk that cannot be cited
    back to its source.
    """


# One INSERT row for the chunks table: text/int columns, a date, and None for
# the nullable section/page/span columns.
_ChunkRow = dict[str, str | int | date | None]


def _is_blank(value: str | None) -> bool:
    """True when a required text field is absent or only whitespace."""
    return value is None or not value.strip()


def validate_chunk_provenance(chunk: Chunk) -> None:
    """Enforce the per-type provenance contract; raise ProvenanceError if broken.

    The chunks DDL (migration 0001_chunks) makes source_id/document_id/text NOT
    NULL but leaves section/page/span nullable, because the requirement is
    source-type-specific and is enforced HERE (one place), not in the schema:

    - Universal: source_id, document_id, span and text must be present.
    - Tabular chunk (no page): must carry a section (its table + band).
    - Document chunk (page set): the page must be a real 1-based number.

    Surfacing these as a clear ProvenanceError up front beats an opaque database
    IntegrityError at INSERT, and guarantees citation resolution (H1-19) always
    has the coordinates it needs.

    Two deliberate scope limits: (1) the source type is INFERRED from page (None
    -> tabular), so this gate cannot require page-presence for a document chunk
    that lost its page — when the H1-08 document producer lands it must guarantee
    a page, or Chunk should gain an explicit kind discriminator (seeded in the
    inbox). (2) Non-provenance column invariants (token_count >= 0, the NOT NULL
    columns) are the DDL's job (migration 0001_chunks); this gate is scoped to
    citation provenance.
    """
    if _is_blank(chunk.source_id):
        raise ProvenanceError("chunk has no source_id (the citation root)")
    if _is_blank(chunk.document_id):
        raise ProvenanceError(f"chunk for source {chunk.source_id!r} has no document_id")
    label = f"{chunk.source_id}/{chunk.document_id}"  # both parts now known-present
    if _is_blank(chunk.span):
        raise ProvenanceError(f"chunk {label} has no span (the in-source locator)")
    if _is_blank(chunk.text):
        raise ProvenanceError(f"chunk {label} span={chunk.span!r} has empty text")
    if chunk.page is None:
        if _is_blank(chunk.section):
            raise ProvenanceError(
                f"tabular chunk {label} span={chunk.span!r} has no section (tabular chunks require section + span)"
            )
    elif chunk.page < 1:
        raise ProvenanceError(f"document chunk {label} span={chunk.span!r} has a non-positive page ({chunk.page})")


def _chunk_to_row(chunk: Chunk) -> _ChunkRow:
    """Project a validated Chunk onto the chunks-table INSERT columns.

    chunk_id (Identity), ts (generated tsvector) and created_at (server default)
    are server-managed and intentionally omitted (see db.chunks_table).
    """
    return {
        "source_id": chunk.source_id,
        "document_id": chunk.document_id,
        "section": chunk.section,
        "page": chunk.page,
        "span": chunk.span,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "access_date": chunk.access_date,
    }


def write_chunks(
    chunks: list[Chunk],
    *,
    engine: Engine,
    replace: bool = True,
    refresh_documents: set[tuple[str, str]] | None = None,
) -> int:
    """Validate every chunk, then write the batch to the chunks table atomically.

    Fail-closed in two senses: (1) provenance is validated for ALL chunks before
    any row is written, so a single bad chunk aborts ingestion with nothing
    persisted; (2) the write runs in one transaction, so a mid-batch database
    error rolls the whole batch back.

    Idempotent by default (replace=True): existing rows are deleted before the
    insert, so re-running refreshes in place instead of duplicating. The delete
    targets the (source_id, document_id) pairs in THIS batch UNION
    ``refresh_documents`` — pass the full set of documents the run is
    authoritative for (every source it processed) so a document that now renders
    ZERO chunks (e.g. a source that reverted to illustrative, or whose rows all
    failed the honesty allowlist) still has its stale rows pruned, never left to
    be cited as current. Documents NOT in either set are untouched, so a partial
    re-ingest (e.g. tabular-only) leaves the rest intact. Embeddings for deleted
    chunks cascade away (FK ON DELETE CASCADE, migration 0002) and are
    regenerated by the embed step (H1-11). Returns the number of chunks written.
    """
    for chunk in chunks:
        validate_chunk_provenance(chunk)

    rows = [_chunk_to_row(chunk) for chunk in chunks]
    doc_keys = sorted({(chunk.source_id, chunk.document_id) for chunk in chunks} | (refresh_documents or set()))
    if not rows and not (replace and doc_keys):
        logger.warning("write_chunks: nothing to write or refresh (no chunks, no documents to prune)")
        return 0

    with engine.begin() as conn:
        if replace and doc_keys:
            deleted = conn.execute(
                chunks_table.delete().where(
                    sa.tuple_(chunks_table.c.source_id, chunks_table.c.document_id).in_(doc_keys)
                )
            ).rowcount
            if deleted:
                logger.info("write_chunks: replaced %d existing chunk(s) across %d document(s)", deleted, len(doc_keys))
        if rows:
            conn.execute(chunks_table.insert(), rows)
    logger.info("write_chunks: wrote %d chunk(s) across %d document(s)", len(rows), len(doc_keys))
    return len(rows)


def _processed_dir_default() -> Path:
    """The dashboard pipeline's processed-output directory (the tabular sources)."""
    return _find_repo_root() / "projects" / "wealthlens-dashboard" / "data" / "processed"


def ingest_slice(*, engine: Engine | None = None, processed_dir: Path | None = None) -> int:
    """Run the corpus-slice ingestion end-to-end; return the chunks written.

    Collects the tabular-source chunks (H1-07), validates provenance, and writes
    them to the chunks table (the STORED tsvector keeps FTS current with no extra
    step). The PDF document path (H1-08) is not built yet, so today's slice is the
    tabular sources only; dense embeddings are a separate step (H1-11). ``engine``
    defaults to one built from the environment (DATABASE_URL).

    Fails loudly when NONE of the expected source CSVs exist (a fresh checkout
    whose gitignored pipeline outputs were never generated): a silent empty corpus
    would let retrieval/evals run against no evidence. A CSV that is present but
    yields no chunks (e.g. wholly illustrative) is legitimate and does not error —
    its document is still refreshed so stale rows cannot survive.
    """
    # Only dispose an engine WE created — a caller-supplied (shared) engine is the
    # caller's to manage (mirrors retrieval/fts.py::search_fts). Engine creation and
    # dir resolution sit INSIDE the try so the internally-built pool is torn down on
    # every error path: a failing _processed_dir_default(), the no-CSV RuntimeError,
    # or write_chunks raising. The `engine is not None` guard covers the case where
    # engine_from_settings() itself raised (created=True but engine never assigned).
    created = engine is None
    try:
        if engine is None:
            engine = engine_from_settings(load_settings())
        if processed_dir is None:
            processed_dir = _processed_dir_default()

        # The documents this run is authoritative for: every spec whose CSV is on
        # disk (so a present-but-now-empty source still gets its stale rows pruned).
        present_specs = [spec for spec in TABLE_SPECS if (processed_dir / spec.csv_name).is_file()]
        if not present_specs:
            raise RuntimeError(
                f"no processed source CSVs found in {processed_dir} "
                f"(expected any of {[spec.csv_name for spec in TABLE_SPECS]}); "
                "regenerate the pipeline outputs (make pipelines) before ingesting"
            )
        refresh_documents = {(spec.source_id, spec.document_id) for spec in present_specs}

        chunks = collect_tabular_chunks(processed_dir=processed_dir)
        logger.info(
            "collected %d tabular chunk(s) from %d source CSV(s); document corpus pending H1-08 (PDF chunking)",
            len(chunks),
            len(present_specs),
        )
        return write_chunks(chunks, engine=engine, refresh_documents=refresh_documents)
    finally:
        if created and engine is not None:
            engine.dispose()


def main() -> None:
    """CLI entrypoint for `make ingest-slice`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = ingest_slice()
    if written == 0:
        # Missing inputs already raised in ingest_slice; reaching 0 here means the
        # source CSVs are present but every row was skipped (e.g. wholly
        # illustrative data failing the honesty allowlist). Flag it rather than
        # reporting a hollow success.
        logger.warning(
            "ingest-slice wrote 0 chunks: source CSVs are present but every row was skipped "
            "(no official rows passed the honesty allowlist). Check the processed data."
        )
    else:
        logger.info("ingest-slice complete: %d chunk(s) in the corpus", written)

    # Dense embeddings (H1-11) are GATED on OPENAI_API_KEY: without a key the corpus
    # is still fully searchable via FTS (the lexical leg), embeddings just aren't
    # generated, so this fails open to a usable degraded mode rather than erroring.
    # embed_corpus is idempotent (anti-join + ON CONFLICT DO NOTHING), so it embeds
    # exactly the chunks this or a prior run left without a vector.
    if load_settings().openai_api_key:
        embedded = embed_corpus()
        logger.info("ingest-slice: %d new embedding(s) written", embedded)
    else:
        logger.info("ingest-slice: OPENAI_API_KEY not set - skipping embeddings (FTS-only corpus)")


if __name__ == "__main__":
    main()
