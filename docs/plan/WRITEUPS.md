# Hero #1 — Writeups (locked titles, fortnight cadence)

Last updated: 2026-06-11
Cadence: writeup #1 ships with M6 (week 6); #2 and #3 follow at fortnight intervals.
Voice: confident, data-driven, non-partisan, personal where useful. Every
statistic cited.

## 1. Turning 11 UK data pipelines into a RAG corpus with provenance
**Slot:** week 6 (with the M6 launch).
- The starting point: 11 reproducible tabular pipelines (ONS, HMRC, BoE, WID)
  with a YAML source registry — and why "just index the CSVs" is not a corpus.
- Provenance at ingestion time: chunks carry source_id/document/section/page/span
  from day one, because citations reconstructed later are citations you can't trust.
- The first document pipeline: fetching IFS/Resolution Foundation reports with
  licence + access date recorded, vs the manual-transcription convention before it.
- Freezing the slice: why a 3-source corpus you can defend beats a 30-source
  corpus you can't, and how the no-new-sources rule was enforced in CLAUDE.md.
- What official statistics do to a chunker: tables, suppressed cells, multi-year
  waves — concrete examples from WAS and HMRC CGT bands.

## 2. An eval harness that says no: abstention and golden sets over official statistics
**Slot:** week 8 (fortnight after #1).
- Refusal as a product feature: the structured "cannot answer from this corpus"
  response, and why honest abstention is the whole credibility story.
- Human-reviewed golden sets: 50-100 Q/A pairs, why the answers were written by
  a human (me) and what the agent was forbidden from fabricating.
- Deterministic checks vs model-graded metrics: citation resolvability, schema
  validity, correct refusal on out-of-corpus questions — the cheap checks that
  catch the expensive failures.
- RAGAS in practice on a small corpus: which metrics moved, which were noise.
- The eval report as a committed artifact: evals in CI, and what a red eval
  gate actually caught during the build.

## 3. What it costs to answer a question about UK wealth: latency, money, and honesty
**Slot:** week 10 (fortnight after #2).
- The public metrics page: p50/p95 latency and cost per query, measured from
  the query log, published live — numbers most demos hide.
- The hard spend cap: budgets table + middleware + 429 refusal, tested by
  exhausting the budget in CI; why a public demo without a hard cap is a
  liability.
- One thin client seam: every model call through one module, and what that
  buys when a gateway (Hero #2) replaces it as a config flip.
- Cost anatomy of a single query: embedding, retrieval, rerank (flag on/off),
  generation — with real per-stage numbers from Langfuse traces.
- What honesty costs: the latency/cost price of abstention checks and citation
  verification, and why it's worth paying.

## 10 named people (demo send list)

The list itself is maintained privately (it contains third-party contact
information); every send is logged there before and after contact. Shape:
7 practitioners/researchers + 3 journalists/civic-data people who could become
real users of the live product.
