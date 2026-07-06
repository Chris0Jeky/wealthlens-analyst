# Hero #1 corpus candidates — IFS/RF reports (decision-ready for H1-01)

Last updated: 2026-06-11

All facts web-verified 2026-06-11 (PDFs downloaded; page counts exact via
pypdf). **Default: proceed with the recommended set of 5 below unless Chris
objects** — picked per the locked plan's "3-5 IFS / Resolution Foundation
reports", one per theme. H1-01 = tag the three tabular sources + paste the
YAML below into `registries/sources.yml` (refresh `access_date` at fetch time).

## Licence terms (site-wide, checked 2026-06-11)

- **IFS** ([ifs.org.uk/copyright](https://ifs.org.uk/copyright), updated
  16/02/2026): **CC BY-NC 4.0** — share/adapt with attribution
  ("Source: Institute for Fiscal Studies. Licensed under CC BY-NC 4.0.");
  commercial use needs written permission. Compatible with WealthLens
  (open-source, non-commercial, citation-first). Deaton Review OUP version is
  **CC BY** (most permissive).
- **Resolution Foundation** ([CC licence page](https://www.resolutionfoundation.org/privacy/creative-commons-licence/)):
  **CC BY-NC-ND 4.0** — reuse with credit, non-commercial, "text not altered
  and used in full". ⚠️ **The ND clause is the one constraint that matters for
  RAG**: chunked excerpting is arguably adaptation. Handling (record in the
  registry notes + answer composer): RF content is *quoted with citation and
  linked to the full report*, never paraphrased-as-ours; chunks are retrieval
  artifacts, the served answer quotes + cites.

## Recommended set (5)

| # | Report | Theme | Pages | Licence |
|---|---|---|---|---|
| 1 | RF **Intergenerational Audit 2024** (Nov 2024) | intergenerational transfers | 73 | CC BY-NC-ND |
| 2 | IFS **R188 Inheritances and inequality over the life cycle** (2021) | inheritance | 95 | CC BY-NC |
| 3 | RF **Before the fall** (Oct 2025) | wealth distribution (latest WAS) | 61 | CC BY-NC-ND |
| 4 | IFS **Reforming inheritance tax** (Green Budget 2023) | wealth taxation | 58 | CC BY-NC |
| 5 | IFS **Trends in income and wealth inequalities** (Deaton Review, OUP 2024) | long-run framing | 135 | **CC BY** (OUP) |

Also verified but not recommended: RF *ISA ISA Baby* (2023, 38pp — savings
policy, weaker fit) and RF *Peaked Interest?* (2023, 63pp — superseded by
*Before the fall* as the distribution pick).

## Ingestion notes (for H1-06)

- ifs.org.uk **HTML** pages serve a Cloudflare challenge to plain fetchers;
  the **PDF URLs download fine via direct HTTP** — fetch_documents.py must
  target the PDF URLs below, not landing pages.
- The RF Audit PDF URL contains a real typo ("Intergenerationl") — it is
  correct as written.
- The old Deaton chapter landing URL redirects; cite the OUP article as
  canonical (DOI 10.1093/ooec/odad100), ingest the IFS-hosted final PDF.

## Paste-ready `registries/sources.yml` entries (H1-01)

```yaml
  - id: rf-intergenerational-audit-2024
    name: "Resolution Foundation — An intergenerational audit for the UK (2024)"
    url: https://www.resolutionfoundation.org/publications/intergenerational-audit-2024/
    pdf_url: https://www.resolutionfoundation.org/app/uploads/2024/11/Intergenerationl-Audit-2024.pdf
    access_date: 2026-06-11
    format: report
    licence: CC BY-NC-ND 4.0
    update_pattern: annual-ish (latest edition; no 2025 edition exists as of 2026-06-11)
    pipeline: null
    analyst_corpus: true
    notes: >-
      6th edition (Broome, Hale, Slaughter; DOI 10.63492/po886z; 73pp). Special
      focus: gifts, inheritances, intergenerational transfers. ND licence:
      serve as quote-with-citation + link to full report, never paraphrase-as-ours.
      PDF URL typo ("Intergenerationl") is real.

  - id: ifs-r188-inheritances-lifecycle
    name: "IFS R188 — Inheritances and inequality over the life cycle (2021)"
    url: https://ifs.org.uk/publications/inheritances-and-inequality-over-life-cycle-what-will-they-mean-younger-generations
    pdf_url: https://ifs.org.uk/sites/default/files/output_url_files/R188-Inheritances-and-inequality-over-the-lifecycle%252520%2525281%252529.pdf
    access_date: 2026-06-11
    format: report
    licence: CC BY-NC 4.0
    update_pattern: one-off
    pipeline: null
    analyst_corpus: true
    notes: >-
      Bourquin, Joyce, Sturrock (Nuffield-funded; 95pp). Cohort projections of
      inheritance flows and their effect on within/between-generation wealth
      inequality. IFS HTML is Cloudflare-gated; fetch the PDF URL directly.

  - id: rf-before-the-fall-2025
    name: "Resolution Foundation — Before the fall: the distribution of household wealth in Britain (2025)"
    url: https://www.resolutionfoundation.org/publications/before-the-fall/
    pdf_url: https://www.resolutionfoundation.org/app/uploads/2025/10/Before-the-fall.pdf
    access_date: 2026-06-11
    format: report
    licence: CC BY-NC-ND 4.0
    update_pattern: series (4th household-wealth audit)
    pipeline: null
    analyst_corpus: true
    notes: >-
      Broome & Kanabar (DOI 10.63492/pcv793; 61pp). Latest comprehensive wealth
      distribution audit on 2020-22 WAS data. ND licence: quote-with-citation
      handling as above.

  - id: ifs-green-budget-2023-iht
    name: "IFS — Reforming inheritance tax (Green Budget 2023, ch. 7)"
    url: https://ifs.org.uk/publications/reforming-inheritance-tax
    pdf_url: https://ifs.org.uk/sites/default/files/2023-09/Reforming-inheritance-tax-1.pdf
    access_date: 2026-06-11
    format: report
    licence: CC BY-NC 4.0
    update_pattern: one-off
    pipeline: null
    analyst_corpus: true
    notes: >-
      Advani & Sturrock (58pp). Wealth-taxation analysis: IHT coverage, revenue
      path, reliefs/avoidance, distributional impact of reform options.

  - id: ifs-deaton-trends-wealth
    name: "IFS Deaton Review — Trends in income and wealth inequalities (OUP 2024)"
    url: https://academic.oup.com/ooec/article/3/Supplement_1/i103/7477701
    pdf_url: https://ifs.org.uk/sites/default/files/2026-03/Trends-in-income-and-wealth-inequalities-IFS-Deaton-Review-of-Inequality-final.pdf
    access_date: 2026-06-11
    format: report
    licence: CC BY 4.0 (OUP published version)
    update_pattern: one-off
    pipeline: null
    analyst_corpus: true
    notes: >-
      Bourquin, Brewer, Wernham (DOI 10.1093/ooec/odad100; 135pp). Long-run
      wealth-vs-income inequality framing. Cite the OUP article as canonical;
      ingest the IFS-hosted final PDF (old chapter URL redirects).
```

Plus, on the three existing tabular entries (`ons-was-wealth`,
`hmrc-cgt-statistics`, `hmrc-tax-receipts`): add `analyst_corpus: true`.
