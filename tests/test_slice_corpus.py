"""Lock the tabular-chunk format + data-honesty rules (H1-07, ADR 0001 §4).

Mostly pure-Python (no DB, no model calls). The renderer takes already-read
rows; one IO test exercises collect_tabular_chunks against a temp fixture CSV
(never the real gitignored pipeline outputs). Covers: exact chunk format, full
per-type provenance (section + span, no page), units + period in the text,
suppressed cells never fabricated as 0, genuine zeros preserved, the fail-closed
official-only allowlist, span disambiguation, number formatting, and the
CSV-reading path (incl. a BOM).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from wealthlens_analyst.ingest.slice_corpus import (
    TABLE_SPECS,
    Chunk,
    CumulativeColumn,
    CumulativeSpec,
    TableSpec,
    ValueColumn,
    _clean_cumulative_pct,
    _clean_number,
    collect_tabular_chunks,
    render_table_chunks,
)

# The real production specs — testing against these locks the shipped format,
# not a throwaway fixture.
_SPECS = {spec.source_id: spec for spec in TABLE_SPECS}
_WAS = _SPECS["ons-was-wealth"]
_CGT = _SPECS["hmrc-cgt-statistics"]
_RECEIPTS = _SPECS["hmrc-tax-receipts"]


def test_was_chunk_format_is_locked() -> None:
    """The WAS renderer emits the exact citable text + provenance we expect."""
    rows = [
        {"decile": "1st (poorest)", "total_wealth_bn": "13.9"},
        {"decile": "10th (richest)", "total_wealth_bn": "5523.2"},  # matches the real CSV
    ]
    chunks = render_table_chunks(rows, _WAS)

    assert len(chunks) == 2
    first = chunks[0]
    assert isinstance(first, Chunk)
    assert first.source_id == "ons-was-wealth"
    assert first.document_id == "was-total-wealth-by-decile"
    assert first.section == "Total household wealth by decile, Great Britain: 1st (poorest)"
    assert first.span == "decile=1st (poorest)"
    assert first.page is None
    assert first.access_date == date(2026, 5, 30)
    assert first.text == (
        "Total household wealth by decile, Great Britain "
        "(ONS Wealth and Assets Survey, April 2020 to March 2022). "
        "Wealth decile 1st (poorest): aggregate total net wealth £13.9bn."
    )
    assert first.token_count == len(first.text.split()) > 0

    # Thousands separator + decimal preserved; units carried.
    assert "£5,523.2bn" in chunks[1].text


def test_tabular_provenance_invariants_hold_for_every_chunk() -> None:
    """Every tabular chunk satisfies the per-type rule: section + span, no page."""
    rows = [
        {"decile": "5th", "total_wealth_bn": "652.0"},
        {"decile": "6th", "total_wealth_bn": "955.2"},
    ]
    for chunk in render_table_chunks(rows, _WAS):
        assert chunk.source_id
        assert chunk.document_id
        assert chunk.section  # non-empty
        assert chunk.span  # non-empty
        assert chunk.page is None  # tabular sources have no pages
        assert chunk.text
        assert chunk.token_count > 0


def test_suppressed_cell_is_omitted_not_fabricated() -> None:
    """A blank cell is dropped from the text; it never becomes a fabricated 0.

    Mirrors the real HMRC CGT '£3,000+' band, where the taxpayer count and
    taxpayer-share cells are suppressed but the gains figures are present.
    """
    rows = [
        {
            "gain_band": "£3,000+",
            "num_taxpayers_thousands": "",  # suppressed
            "total_gains_millions": "1.0",
            "share_of_gains_pct": "0.0",  # a GENUINE zero
            "share_of_taxpayers_pct": "",  # suppressed
        }
    ]
    (chunk,) = render_table_chunks(rows, _CGT)

    # Present figures appear (genuine zero preserved as "0%").
    assert "total gains in this band £1m" in chunk.text
    assert "share of all taxable gains in this band 0%" in chunk.text
    # Suppressed figures are absent — and never rendered as a fabricated 0.
    assert "taxpayers with gains in this band" not in chunk.text
    assert "share of all CGT taxpayers" not in chunk.text


def test_cgt_without_band_lower_falls_back_to_raw_label() -> None:
    """Honest fallback path: a CGT row missing band_lower keeps its raw '£X+'
    label (never a fabricated range), while per-band figures stay scoped 'in this
    band'. Production rows always carry band_lower and render the explicit range
    instead (see test_cgt_renders_explicit_band_ranges)."""
    rows = [
        {
            "gain_band": "£6,000+",  # no band_lower key -> raw label retained
            "num_taxpayers_thousands": "61.0",
            "total_gains_millions": "461.0",
            "share_of_gains_pct": "0.7",
            "share_of_taxpayers_pct": "17.0",
        }
    ]
    (chunk,) = render_table_chunks(rows, _CGT)
    assert "Size-of-gain band £6,000+:" in chunk.text
    assert "share of all taxable gains in this band 0.7%" in chunk.text
    assert "taxpayers with gains in this band 61 thousand" in chunk.text


def test_row_with_every_value_missing_yields_no_chunk() -> None:
    """If a band has no usable figures at all, it produces no chunk (no empty cite)."""
    rows = [
        {
            "gain_band": "£0+",
            "num_taxpayers_thousands": "",
            "total_gains_millions": "",
            "share_of_gains_pct": "",
            "share_of_taxpayers_pct": "",
        }
    ]
    assert render_table_chunks(rows, _CGT) == []


def test_blank_band_label_is_skipped() -> None:
    """Rows with no section label (e.g. spreadsheet footers) are not chunked."""
    rows = [
        {"decile": "", "total_wealth_bn": "1.0"},
        {"decile": "   ", "total_wealth_bn": "2.0"},
        {"decile": "1st (poorest)", "total_wealth_bn": "13.9"},
    ]
    chunks = render_table_chunks(rows, _WAS)
    assert len(chunks) == 1
    assert chunks[0].span == "decile=1st (poorest)"


def test_span_is_disambiguated_on_duplicate_band_label() -> None:
    """A repeated band label never yields two chunks with the same span."""
    rows = [
        {"decile": "5th", "total_wealth_bn": "1.0"},
        {"decile": "5th", "total_wealth_bn": "2.0"},
    ]
    chunks = render_table_chunks(rows, _WAS)
    spans = [c.span for c in chunks]
    assert spans == ["decile=5th", "decile=5th#2"]
    assert len(set(spans)) == len(spans)


def test_honesty_filter_is_a_fail_closed_allowlist() -> None:
    """Only known-official data_source rows are ingested; everything else skipped."""
    base = {
        "income_tax_bn": "270.0",
        "nics_bn": "180.0",
        "cgt_bn": "15.0",
        "iht_bn": "7.5",
        "sdlt_bn": "12.0",
    }
    rows = [
        {"year": "2023-24", **base, "data_source": "live"},  # official -> ingested
        {"year": "2022-23", **base, "data_source": "illustrative"},  # skipped
        {"year": "2021-22", **base, "data_source": "estimated"},  # unknown marker -> skipped
        {"year": "2020-21", **base, "data_source": ""},  # blank -> skipped
        {"year": "2019-20", **base},  # missing column -> skipped
    ]
    chunks = render_table_chunks(rows, _RECEIPTS)

    assert len(chunks) == 1
    assert chunks[0].span == "year=2023-24"
    assert "Income Tax receipts £270bn" in chunks[0].text
    assert "Capital Gains Tax receipts £15bn" in chunks[0].text
    # The editorial work/wealth split is not ingested under the HMRC attribution.
    assert "share from taxes" not in chunks[0].text


def test_real_tax_composition_is_all_illustrative_today() -> None:
    """The shipped tax_composition data is illustrative, so it yields no chunks.

    Guards the mission rule: the analyst must not cite fabricated figures as
    official HMRC receipts. When the pipeline emits real ("live") data this
    test updates.
    """
    rows = [
        {"year": "2018-19", "income_tax_bn": "191.0", "data_source": "illustrative"},
        {"year": "2023-24", "income_tax_bn": "270.0", "data_source": "illustrative"},
    ]
    assert render_table_chunks(rows, _RECEIPTS) == []


def test_clean_number_formatting_and_missing() -> None:
    """Number cleaning: noise rounded, thousands grouped/tolerated, missing -> None."""
    assert _clean_number("13.9") == "13.9"
    assert _clean_number("5523.0") == "5,523"  # integral -> grouped int
    assert _clean_number("5523.2") == "5,523.2"  # grouped + decimal
    assert _clean_number("100.10000000000001") == "100.1"  # float-repr noise rounded
    assert _clean_number("1,234.5") == "1,234.5"  # thousands separator in input tolerated
    assert _clean_number("0.0") == "0"  # genuine zero preserved
    assert _clean_number("0.0000004") not in (None, "0")  # tiny non-zero NOT collapsed to 0
    assert _clean_number("") is None  # blank -> omit
    assert _clean_number("   ") is None
    assert _clean_number(None) is None
    assert _clean_number("n/a") is None  # non-numeric -> omit
    assert _clean_number("inf") is None  # non-finite -> omit


def test_collect_tabular_chunks_reads_csv_with_bom_and_skips_missing(tmp_path: Path) -> None:
    """The IO path reads a (BOM-prefixed) CSV and skips absent sources cleanly."""
    # Only the WAS CSV exists; written utf-8-sig so it carries a BOM (as a
    # spreadsheet export would). The CGT/receipts CSVs are absent -> skipped.
    csv_path = tmp_path / "ons_wealth_by_decile.csv"
    csv_path.write_text("decile,total_wealth_bn\n1st (poorest),13.9\n10th (richest),5523.2\n", encoding="utf-8-sig")

    chunks = collect_tabular_chunks(processed_dir=tmp_path)

    # BOM did not corrupt the 'decile' header (else 0 chunks); both rows rendered.
    assert len(chunks) == 2
    assert {c.source_id for c in chunks} == {"ons-was-wealth"}
    assert chunks[0].span == "decile=1st (poorest)"
    assert "£5,523.2bn" in chunks[1].text


def test_table_specs_are_wellformed() -> None:
    """Each shipped spec is usable: unique ids, real CSV names, value columns."""
    source_ids = [spec.source_id for spec in TABLE_SPECS]
    assert len(source_ids) == len(set(source_ids))  # unique
    for spec in TABLE_SPECS:
        assert spec.csv_name.endswith(".csv")
        assert spec.value_columns  # at least one statistic to render
        assert spec.section_column
        assert isinstance(spec.access_date, date)
        # A table with an honesty flag must define a non-empty official allowlist.
        if spec.honesty_flag_column is not None:
            assert spec.official_markers
        # A cumulative clause is anchored on the band's lower bound.
        if spec.cumulative is not None:
            assert spec.band_lower_column is not None
            assert spec.cumulative.columns  # at least one cumulative column to render


# --- CGT size-band enrichments: explicit ranges + cumulative concentration -----
# (H1-07 follow-up; serves golden G-007 concentration / G-008 share above £1m.)

# Four real, CONSECUTIVE HMRC bands so the rendered ranges match the source table.
# Values verbatim from projects/.../data/processed/hmrc_cgt_concentration.csv.
_CGT_TOP_BANDS = [
    {
        "gain_band": "£500,000+",
        "band_lower": "500000.0",
        "num_taxpayers_thousands": "8.0",
        "total_gains_millions": "5705.0",
        "share_of_gains_pct": "9.1",
        "share_of_taxpayers_pct": "2.2",
        "cumul_gains_from_top_pct": "70.0",
        "cumul_taxpayers_from_top_pct": "5.0",
    },
    {
        "gain_band": "£1,000,000+",
        "band_lower": "1000000.0",
        "num_taxpayers_thousands": "5.0",
        "total_gains_millions": "6390.0",
        "share_of_gains_pct": "10.2",
        "share_of_taxpayers_pct": "1.4",
        "cumul_gains_from_top_pct": "60.900000000000006",
        "cumul_taxpayers_from_top_pct": "2.8",
    },
    {
        "gain_band": "£2,000,000+",
        "band_lower": "2000000.0",
        "num_taxpayers_thousands": "3.0",
        "total_gains_millions": "9189.0",
        "share_of_gains_pct": "14.6",
        "share_of_taxpayers_pct": "0.8",
        "cumul_gains_from_top_pct": "50.7",
        "cumul_taxpayers_from_top_pct": "1.4",
    },
    {
        "gain_band": "£5,000,000+",
        "band_lower": "5000000.0",
        "num_taxpayers_thousands": "2.0",
        "total_gains_millions": "22714.0",
        "share_of_gains_pct": "36.1",
        "share_of_taxpayers_pct": "0.6",
        "cumul_gains_from_top_pct": "36.1",
        "cumul_taxpayers_from_top_pct": "0.6",
    },
]


def test_cgt_renders_explicit_band_ranges() -> None:
    """A '£X+' band is rendered as a half-open range (top band: 'and above')."""
    chunks = {c.span: c for c in render_table_chunks(_CGT_TOP_BANDS, _CGT)}

    # Middle bands read as [lower, next-lower) -> "to under"; span still pins the
    # RAW HMRC label; per-band figures stay scoped "in this band".
    mid = chunks["gain_band=£1,000,000+"]
    assert "Size-of-gain band £1,000,000 to under £2,000,000:" in mid.text
    assert mid.section == "Capital Gains Tax by size of gain: £1,000,000 to under £2,000,000"
    assert "share of all taxable gains in this band 10.2%" in mid.text
    assert "Size-of-gain band £500,000 to under £1,000,000:" in chunks["gain_band=£500,000+"].text

    # The highest band is open-ended.
    top = chunks["gain_band=£5,000,000+"]
    assert "Size-of-gain band £5,000,000 and above:" in top.text
    assert top.section == "Capital Gains Tax by size of gain: £5,000,000 and above"


def test_cgt_cumulative_concentration_clause_serves_g008() -> None:
    """Each band carries the cumulative-from-the-top concentration figure.

    The £1,000,000+ chunk states the share of all gains going to gains above £1m
    (golden G-008) directly and citably. The subject is the taxpayers, so the
    gains share and the headcount share each read with correct grammar (never
    "gains ... account for ... taxpayers").
    """
    chunks = {c.span: c for c in render_table_chunks(_CGT_TOP_BANDS, _CGT)}

    assert (
        "Cumulatively, taxpayers with gains of £1,000,000 and above "
        "accounted for 60.9% of all taxable gains and made up 2.8% of all CGT taxpayers."
    ) in chunks["gain_band=£1,000,000+"].text
    assert (
        "Cumulatively, taxpayers with gains of £5,000,000 and above "
        "accounted for 36.1% of all taxable gains and made up 0.6% of all CGT taxpayers."
    ) in chunks["gain_band=£5,000,000+"].text


def test_cgt_cumulative_rounding_overflow_clamped_and_suppressed_omitted() -> None:
    """The low bands' >100% cumulative (a rounding artefact) clamps to 100%, and a
    suppressed cumulative cell is omitted, never fabricated."""
    rows = [
        {  # real £0+ row: both cumulatives overshoot 100 from 1-dp cumsum rounding
            "gain_band": "£0+",
            "band_lower": "0.0",
            "total_gains_millions": "1.0",
            "share_of_gains_pct": "0.0",
            "cumul_gains_from_top_pct": "100.10000000000001",
            "cumul_taxpayers_from_top_pct": "100.6",
        },
        {  # real £3,000+ row: taxpayer cumulative suppressed by HMRC
            "gain_band": "£3,000+",
            "band_lower": "3000.0",
            "total_gains_millions": "1.0",
            "share_of_gains_pct": "0.0",
            "cumul_gains_from_top_pct": "100.10000000000001",
            "cumul_taxpayers_from_top_pct": "",
        },
    ]
    chunks = {c.span: c for c in render_table_chunks(rows, _CGT)}

    # Both overshoots clamp to exactly 100% (no fabricated 100.1).
    assert (
        "taxpayers with gains of £0 and above accounted for 100% of all taxable gains "
        "and made up 100% of all CGT taxpayers."
    ) in chunks["gain_band=£0+"].text
    assert "100.1" not in chunks["gain_band=£0+"].text

    # Suppressed taxpayer cumulative is dropped; the clause keeps only the gains share.
    band_3k = chunks["gain_band=£3,000+"].text
    assert "taxpayers with gains of £3,000 and above accounted for 100% of all taxable gains." in band_3k
    assert "CGT taxpayers" not in band_3k  # neither per-band nor cumulative fabricated


def test_clean_cumulative_pct_clamps_overflow_and_omits_out_of_tolerance() -> None:
    """Clamp the rounding overflow to the ceiling; omit a genuine anomaly."""
    assert _clean_cumulative_pct("100.6", 100.0, 1.0) == "100"  # rounding artefact -> clamp
    assert _clean_cumulative_pct("100.10000000000001", 100.0, 1.0) == "100"
    assert _clean_cumulative_pct("100.0", 100.0, 1.0) == "100"  # exactly the ceiling
    assert _clean_cumulative_pct("60.9", 100.0, 1.0) == "60.9"  # in range untouched
    assert _clean_cumulative_pct("101.5", 100.0, 1.0) is None  # beyond tolerance -> omit
    assert _clean_cumulative_pct("-5", 100.0, 1.0) is None  # negative is impossible -> omit
    assert _clean_cumulative_pct("", 100.0, 1.0) is None  # blank -> omit
    assert _clean_cumulative_pct(None, 100.0, 1.0) is None
    assert _clean_cumulative_pct("n/a", 100.0, 1.0) is None  # non-numeric -> omit


def test_real_cgt_spec_enables_ranges_and_cumulative() -> None:
    """The shipped CGT spec is configured for both enrichments (locks production)."""
    assert _CGT.band_lower_column == "band_lower"
    assert _CGT.cumulative is not None
    assert _CGT.cumulative.subject == "taxpayers with gains of"
    assert {(c.column, c.verb) for c in _CGT.cumulative.columns} == {
        ("cumul_gains_from_top_pct", "accounted for"),
        ("cumul_taxpayers_from_top_pct", "made up"),
    }
    # WAS and receipts are wholly categorical: neither enrichment applies.
    assert _WAS.band_lower_column is None and _WAS.cumulative is None
    assert _RECEIPTS.band_lower_column is None and _RECEIPTS.cumulative is None


def test_cumulative_spec_requires_a_band_lower_column() -> None:
    """A cumulative clause without a lower-bound column is a construction error."""
    with pytest.raises(ValueError, match="band_lower_column"):
        TableSpec(
            source_id="bad",
            csv_name="bad.csv",
            document_id="d",
            table_label="T",
            period="P",
            section_column="band",
            section_noun="band",
            value_columns=(ValueColumn("v", "label"),),
            access_date=date(2026, 1, 1),
            cumulative=CumulativeSpec(subject="x of", columns=(CumulativeColumn("c", "did", "of all x"),)),
        )
