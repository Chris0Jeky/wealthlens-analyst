# CLAUDE.md — WealthLens Analyst (Hero #1)

Product memory for `projects/wealthlens-analyst/`. Loaded automatically when
working in this subtree; travels with the product when it is extracted to its
own repo. Repo-wide rules in the root `CLAUDE.md` / `AGENTS.md` still apply.

## Mission

Evidence-backed research analyst over official UK wealth statistics.
Citation-first retrieval, honest abstention, a committed eval harness, visible
latency/cost numbers, a hard spend cap.
Positioning: "I make LLM systems cheap, reliable, and provably valuable in production."

## The plan is FINAL

`docs/plan/HERO1_PLAN.md` (milestones M0-M6) and `tasks/hero1-backlog.md`
(ordered half-day tasks) are locked. Do not re-plan, re-architect, or re-sequence.
Anything genuinely open lives in ADR 0003 and is **Chris's decision, not
yours** (exception: on 2026-06-11 Chris explicitly delegated D1/D2/D4, now
recorded in the ADR's decision record; D3 hosting remains his).

## Locked decisions (compressed — full text in docs/adr/)

- **Corpus slice, FROZEN until v1 ships:** ONS Wealth and Assets Survey, HMRC
  distributional statistics, 3-5 IFS/Resolution Foundation reports. Reuse
  `registries/sources.yml` + existing pipeline conventions. **Adding any other
  source before the live URL exists is forbidden.**
- **Retrieval:** Postgres FTS + pgvector, fused with RRF (k=60); reranker
  behind `RERANK_ENABLED`, default OFF. (ADR 0001)
- **Citations:** chunk-level; provenance columns (source_id, document_id,
  section, page, span) captured at ingestion, never reconstructed.
- **Abstention:** confidence gate returning a structured "cannot answer from
  this corpus" refusal. Refusal is a product feature, not an error path.
- **Evals:** 50-100 HUMAN-reviewed golden Q/A pairs + RAGAS + deterministic
  checks (citation resolvability, schema validity, correct refusal,
  latency/cost bounds). **Never fabricate golden answers — Chris writes them.**
- **Observability:** OTel traces to self-hosted Langfuse; public metrics page
  (p50/p95 latency, cost/query).
- **Cost:** hard spend cap in-app (budgets table + middleware → 429/refusal),
  fail-closed. Every model call goes through `src/wealthlens_analyst/llm/client.py`
  — **no other module may import a provider SDK**. (ADR 0002)
- **Stack:** Python 3.11+, FastAPI, Postgres+pgvector, Alembic, pytest. Match
  monorepo ruff/mypy/CI conventions; mypy strict for this package.
- **Shipped means:** live URL + committed eval report + writeup #1 published +
  demo sent to 10 named people. Nothing else counts as done.

## Build order

M0 kickoff → M1 ingest (slice → chunks with provenance, FTS, embeddings) →
M2 hybrid retrieval behind /ask (debug mode) → M3 reranker + citations →
M4 abstention → M5 RAGAS + Langfuse + spend cap + metrics page → M6 live URL,
README failure modes, writeup #1, demo sends. Acceptance criteria per
milestone: `docs/plan/HERO1_PLAN.md`.

## Subtree map

```
src/wealthlens_analyst/
  config.py   settings from env (DATABASE_URL, model ids, spend cap)
  db.py       SQLAlchemy engine factory + chunks Core table (H1-09 write path)
  retrieval/  fts.py dense.py fuse_rrf.py rerank.py   # ADR 0001
  answer/     compose.py citations.py abstain.py
  llm/        client.py                               # THE seam (ADR 0002)
  budget/     models.py middleware.py                 # hard cap (ADR 0002)
  api/        app.py routes.py                        # /ask /healthz /metrics/data
  ingest/     slice_corpus.py fetch_documents.py
migrations/   Alembic (hand-written revisions in versions/)
evals/        golden/ checks/ run_ragas.py reports/
tests/
```

## Key commands (root Makefile)

`make dev` (uvicorn) · `make ingest-slice` · `make eval-golden-validate` ·
`make eval-deterministic` · `make eval-ragas` · `make eval-report`

## Engineering cap

No new test infrastructure beyond what the evals need. No speculative
abstractions. One CI job (`ci-analyst.yml`). Shipping beats polish.

## NEVER DO

- Re-plan, re-architect, or propose alternative frameworks/corpora/sequencing.
- Add a corpus source before the live URL exists.
- Fabricate golden answers, statistics, citations, or ground truth of any kind.
- Add test infrastructure beyond what the evals need.
- Call a provider SDK outside `llm/client.py`.
- Commit secrets (keys go in `.env`, documented in `.env.example` only).
- Modify `strategy/`, `vision/`, or `legal/` docs, or reintroduce the
  personal-material directories that moved to the private `hq-private` repo
  on 2026-06-13 (`identity/`, `journal/`, `meetings/`, `people/`,
  `tasks/applications/`, `tasks/outreach/`, `.codex/memories/`).
- Skip the spend-cap path for "internal" calls — every model call is metered.
