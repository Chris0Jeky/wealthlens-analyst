# Hero #1 — Writeups

Last updated: 2026-07-06 (re-sequenced at extraction: the cost teardown leads,
the corpus-provenance piece becomes its sidebar, and the abstention
calibration ships first as a micro-note the moment its numbers exist).
Voice: confident, data-driven, non-partisan, personal where useful. Every
statistic cited. Every piece leads with a measured number or a reproducible
proof, never a capability claim.

## 0. Micro-note: How I calibrated my RAG's refusals

**Slot:** as soon as the M5 calibration report is committed (400-800 words).
- The two-distribution plot (answerable vs out-of-corpus signal) and how the
  threshold pair was chosen — the plot doubles as the README hero image.
- Results as counts ("14/15 out-of-corpus refused at 3/50 false refusals");
  rates only when n >= 50, and say why ("I report counts because the n is
  small").
- The near-miss out-of-corpus class ("median household wealth in Germany"
  against a UK corpus) and what it did to the thresholds.
- Why refusals cost retrieval only — zero generation spend, shown from real
  query_log rows.
- Community share ends with a question: "What OOC refusal recall do you
  target before shipping RAG?"

## 1. What it costs to answer a question about UK wealth: a teardown

**Slot:** ships with M6 (the launch writeup). 1200-1600 words.
- The per-query figure, first sentence; then the decomposition: embedding
  amortization, retrieval (effectively free — say why), rerank (the flag and
  its price), generation (the dominant term), the judge-cost surprise from
  RAGAS runs.
- What teams never measure: abstention's cost profile (refusals cost
  retrieval only; link micro-note 0), cache effects preview.
- The spend cap: estimate-reserve-reconcile, fail-closed, why a 429 beats a
  silent degrade.
- Sidebar — turning reproducible UK data pipelines into a corpus with
  provenance: the source registry, chunk-level provenance captured at
  ingestion (citations reconstructed later are citations you can't trust),
  what official statistics do to a chunker (tables, suppressed cells,
  multi-year waves), freezing an 8-source slice you can defend, and the OGL
  v3 / CC licensing discipline.
- Reproduce pointer, the live metrics page, and a "try it yourself" link —
  the demo is the distribution.

## 2. An eval harness that says no: abstention and golden sets over official statistics

**Slot:** fortnight after #1.
- Refusal as a product feature: the structured "cannot answer from this
  corpus" response, and why honest abstention is the whole credibility story.
- Human-reviewed golden sets: 50-100 Q/A pairs, why the answers were written
  by a human and what the agent was forbidden from fabricating.
- Deterministic checks vs model-graded metrics: citation resolvability,
  schema validity, correct refusal on out-of-corpus questions — the cheap
  checks that catch the expensive failures.
- RAGAS in practice on a small corpus: which metrics moved, which were noise.
- The eval report as a committed artifact: evals in CI, and what a red eval
  gate actually caught during the build.

## 10 named people (demo send list)

The list itself is maintained privately (it contains third-party contact
information); every send is logged there before and after contact. Shape:
7 practitioners/researchers + 3 journalists/civic-data people who could
become real users of the live product.
