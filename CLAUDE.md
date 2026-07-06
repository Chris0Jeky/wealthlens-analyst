# CLAUDE.md — WealthLens Analyst (Hero #1)

Product memory and agent contract for this repo. Extracted from the
`wealthlens-hq` monorepo on 2026-07-06 (the import commit records the source
sha); the plan, backlog, and ADRs travelled with it.

## Mission

Evidence-backed research analyst over official UK wealth statistics.
Citation-first retrieval, honest abstention, a committed eval harness, visible
latency/cost numbers, a hard spend cap.
Positioning: "Every number cited, every refusal calibrated, every query costed."
(Supporting line: "I make LLM systems cheap, reliable, and provably valuable
in production.")

## The plan is FINAL

`docs/plan/HERO1_PLAN.md` (milestones M0-M6) and `tasks/hero1-backlog.md`
(ordered half-day tasks) are locked. Do not re-plan, re-architect, or re-sequence.
Anything genuinely open lives in ADR 0003 and is **Chris's decision, not
yours** (exception: on 2026-06-11 Chris explicitly delegated D1/D2/D4, now
recorded in the ADR's decision record; D3 hosting remains his).

## Locked decisions (compressed — full text in docs/adr/)

- **Corpus slice, FROZEN until v1 ships:** ONS Wealth and Assets Survey, HMRC
  distributional statistics, 3-5 IFS/Resolution Foundation reports. Corpus
  identity lives in `registries/sources.yml` (the trimmed frozen-corpus copy
  carried in the extraction; 8 entries) and `data/corpus_manifest.yml`.
  **Adding any other source before the live URL exists is forbidden.**
- **Retrieval:** Postgres FTS + pgvector, fused with RRF (k=60); reranker
  behind `RERANK_ENABLED`, default OFF. (ADR 0001)
- **Citations:** chunk-level; provenance columns (source_id, document_id,
  section, page, span) captured at ingestion, never reconstructed.
- **Abstention:** confidence gate returning a structured "cannot answer from
  this corpus" refusal. Refusal is a product feature, not an error path.
- **Evals:** 50-100 HUMAN-reviewed golden Q/A pairs + RAGAS + deterministic
  checks (citation resolvability, schema validity, correct refusal,
  latency/cost bounds). **Never fabricate golden answers — Chris writes them.**
- **Observability:** structured JSONL request/eval logging + the public
  metrics page (p50/p95 latency, cost/query) for v1; the full tracing stack
  arrives with the gateway that fronts this service later.
- **Cost:** hard spend cap in-app (budgets table + middleware → 429/refusal),
  fail-closed. Every model call goes through `src/wealthlens_analyst/llm/client.py`
  — **no other module may import a provider SDK**. (ADR 0002)
- **Stack:** Python 3.11+, FastAPI, Postgres+pgvector, Alembic, pytest;
  ruff + strict mypy for this package (config in `pyproject.toml`).
- **Shipped means:** live URL + committed eval report + writeup #1 published +
  demo sent to 10 named people. Nothing else counts as done.

## Build order

M0 kickoff → M1 ingest (slice → chunks with provenance, FTS, embeddings) →
M2 hybrid retrieval behind /ask (debug mode) → M3 reranker + citations →
M4 abstention → M5 RAGAS + spend cap + metrics page → M6 live URL,
README failure modes, writeup #1, demo sends. Acceptance criteria per
milestone: `docs/plan/HERO1_PLAN.md`. Status at extraction: M0-M4 done.

## Repo map

```
src/wealthlens_analyst/
  config.py   settings from env (DATABASE_URL, model ids, spend cap)
  db.py       SQLAlchemy engine factory + chunks Core table (H1-09 write path)
  retrieval/  fts.py dense.py fuse_rrf.py rerank.py   # ADR 0001
  answer/     compose.py citations.py abstain.py
  llm/        client.py                               # THE seam (ADR 0002)
  budget/     models.py middleware.py                 # hard cap (ADR 0002)
  api/        app.py routes.py schemas.py             # /ask /healthz /metrics/data
  ingest/     slice_corpus.py fetch_documents.py
migrations/   Alembic (hand-written revisions in versions/)
evals/        golden/ checks/ run_ragas.py reports/
registries/   sources.yml (frozen-corpus registry, trimmed copy)
data/         corpus_manifest.yml (committed) · raw/ processed/ corpus/ (gitignored)
scripts/      fetch_corpus.py
docs/plan/    HERO1_PLAN.md WRITEUPS.md · docs/adr/ 0001-0003
tasks/        hero1-backlog.md hero1-corpus-candidates.md
tests/
```

## Key commands (Makefile)

`make db-up` (Postgres+pgvector on :15432) · `make dev` (uvicorn :8100) ·
`make ingest-slice` · `make test` · `make lint` · `make eval-golden-validate` ·
`make eval-deterministic` · `make eval-ragas` · `make eval-report`

## Engineering cap

No new test infrastructure beyond what the evals need. No speculative
abstractions. One CI job (`.github/workflows/ci.yml`). Shipping beats polish.

## NEVER DO

- Re-plan, re-architect, or propose alternative frameworks/corpora/sequencing.
- Add a corpus source before the live URL exists.
- Fabricate golden answers, statistics, citations, or ground truth of any kind.
- Add test infrastructure beyond what the evals need.
- Call a provider SDK outside `llm/client.py`.
- Commit secrets (keys go in `.env`, documented in `.env.example` only).
- Add personal or planning material that belongs in the maintainer's private
  repos — this repo is the product only (code, plan, ADRs, evals).
- Skip the spend-cap path for "internal" calls — every model call is metered.
