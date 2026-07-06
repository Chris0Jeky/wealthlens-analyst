# WealthLens Analyst

> Every number cited, every refusal calibrated, every query costed.

Evidence-backed research analyst over official UK wealth statistics — the ONS
Wealth and Assets Survey, HMRC distributional statistics, and selected
IFS/Resolution Foundation reports. Citation-first hybrid RAG with honest
abstention, a committed eval harness, and a hard, fail-closed spend cap.

Extracted from the [`wealthlens-hq`](https://github.com/Chris0Jeky/wealthlens-hq)
monorepo on 2026-07-06 (the import commit records the source revision); the
plan, backlog, and ADRs travelled with the code.

## Numbers pending: measured, not promised

The eval suite is live (deterministic checks gate every merge in CI). The
first public numbers — per-query cost decomposed, p50/p95 latency, abstention
calibration reported as counts — land with the committed M5 eval report and
the public metrics page. Until they exist, this README leads with this block,
never a capability claim.

**Scale disclosure:** a single-tenant system built with production
disciplines (fail-closed budgets, provenance-first ingestion, eval gates in
CI). Measured numbers describe this workload, nothing larger.

## Status

**M0-M4 done · M5 (evals, spend cap, metrics page) in progress.**
`POST /ask` answers with hybrid retrieval (Postgres FTS + pgvector dense,
RRF-fused) → an abstention gate ahead of generation (weak evidence refuses
before any generation spend) → cited generation through one client seam →
citation resolution against the frozen source registry. Serving policy: an
answer is served only when EVERY citation resolves; if any cited id is
fabricated or unverifiable the request is refused — a citation-first product
never serves an uncited claim. `POST /ask?debug=retrieval` returns the fused
candidate list (component ranks, no generation). Every request writes a
query_log accounting row (embed + generation spend, latency, decision).

Plan: `docs/plan/HERO1_PLAN.md` · backlog: `tasks/hero1-backlog.md` ·
decisions: `docs/adr/0001-0003` · agent contract: `CLAUDE.md`.

## Architecture

```
question ──▶ POST /ask
  ├─ hybrid retrieval: Postgres FTS ∥ pgvector dense ──▶ RRF fusion (k=60)
  ├─ reranker behind RERANK_ENABLED (default OFF — flips only on measured evidence)
  ├─ abstention gate (fused-score threshold + min-hits)
  │     └─ weak evidence ──▶ structured refusal, reason enum, ZERO generation spend
  ├─ generation via llm/client.py (THE seam — no other module imports a provider SDK)
  ├─ citation resolution vs the frozen registry ──▶ serve only if every citation resolves
  └─ query_log row (tokens, cost, latency, decision) ──▶ metrics page (M5)
  hard monthly spend cap: estimate-reserve-reconcile, fail-closed 429 (M5)
```

## Quick start

```bash
make db-up            # Postgres+pgvector on localhost:15432
make install          # pip install -e ".[dev,evals]"
cp .env.example .env  # fill in, then export: set -a; . ./.env; set +a
alembic upgrade head
make ingest-slice     # chunk -> provenance gate -> write; embeds if OPENAI_API_KEY set
make dev              # uvicorn on 127.0.0.1:8100

# A cited answer or an honest refusal (spends one generation):
curl -X POST "http://127.0.0.1:8100/ask" \
  -H "Content-Type: application/json" -d '{"question": "who holds the most wealth?"}'
# Retrieval diagnostics only (no generation):
curl -X POST "http://127.0.0.1:8100/ask?debug=retrieval" \
  -H "Content-Type: application/json" -d '{"question": "who holds the most wealth?"}'

make test && make lint && make eval-deterministic
```

Note: the tabular corpus CSVs are currently produced by the upstream
`wealthlens-hq` data pipelines and placed in `data/processed/` — the
self-contained fetch is a seeded follow-up (`tasks/hero1-backlog.md`,
standalone-repo section).

## Corpus and licensing

The corpus is a **frozen 8-source slice** (`registries/sources.yml`,
`data/corpus_manifest.yml`); adding a source before the live URL exists is
forbidden by the locked plan. ONS and HMRC statistics are used under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/);
IFS and Resolution Foundation reports carry their publishers' CC licences
(BY / BY-NC / BY-NC-ND, per manifest entry) — fetched for analysis, cited
with provenance, never redistributed here. Code is MIT (`LICENSE`).

## Failure modes

_To be written at M6 (see `docs/plan/HERO1_PLAN.md`): what the system refuses
to answer and why, known weak spots (degraded table extraction, weak query
classes), corpus staleness date, and honest limitations._

## How this was built

Agentic coding under a locked plan: agents draft; the maintainer owns every
golden answer, ADR decision, and threshold; deterministic eval checks gate
every merge in CI; ground truth is never fabricated — a test enforces that
unreviewed golden items stay empty. The plan, the ordered backlog, and the
ADRs are all in this repo: the process is part of the product.
