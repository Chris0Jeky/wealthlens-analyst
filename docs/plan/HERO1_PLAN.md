# Hero #1 — WealthLens v2: Evidence-Backed Research Analyst (RAG)

Last updated: 2026-06-11
Status: **locked plan** — do not re-architect. Open decisions live in ADR 0003 only.
Backlog: `tasks/hero1-backlog.md` · ADRs: `docs/adr/0001..0003` · Product memory: `projects/wealthlens-analyst/CLAUDE.md`

**Positioning:** "I make LLM systems cheap, reliable, and provably valuable in production."

**Shipped means (the only definition of done):** live URL + eval report committed +
writeup #1 published + demo link sent to 10 named people. Nothing else counts.

## Corpus slice (FROZEN until v1 ships)

ONS Wealth and Assets Survey + HMRC distributional statistics (CGT statistics,
tax receipts) + 3-5 IFS / Resolution Foundation reports. All registered in
`registries/sources.yml` with URL, access date, format, licence. **Adding any
other source before the live URL exists is forbidden.**

## Milestones and acceptance criteria

### M0 — Kickoff (week 1) — this session
- [ ] Branch `chore/hero1-kickoff` carries: this plan, `tasks/hero1-backlog.md`
      (25-40 half-day tasks; the first ten blocked on nothing outside the
      backlog itself — H1-05..10 chain by design), ADRs 0001-0003, product
      scaffolding under `projects/wealthlens-analyst/` (stubs import cleanly),
      eval skeleton with 20 DRAFT golden questions, Makefile + CI extensions,
      `.env.example`.
- [ ] Corpus slice tagged in `registries/sources.yml` (`analyst_corpus: true`
      on exactly the slice sources; IFS/RF report entries added with licence).
- [ ] Alembic migration drafts exist for `chunks` (provenance columns:
      source_id, document_id, section, page, span), `embeddings` (pgvector
      column + index), `budgets`, `query_log` (tokens, cost, latency, decision).
- [ ] Golden schema + 20 DRAFT questions (15 in-corpus, 5 out-of-corpus) ready
      for Chris's weekend review. **Golden answers are written by Chris, never
      fabricated.**

### M1 — Corpus ingested (weeks 1-2)
- [ ] Postgres + pgvector running locally (docker-compose service); Alembic
      migrations apply cleanly from empty.
- [ ] New document-fetch step downloads the 3-5 IFS/RF reports (PDF) with
      registry-recorded URL + access date + licence (this is the repo's FIRST
      document pipeline — existing 11 pipelines are tabular-only).
- [ ] `ingest/slice_corpus.py` chunks the slice (tabular sources rendered to
      citable text via the existing pipelines' processed outputs; reports via
      PDF extraction) and writes `chunks` rows with ALL provenance columns
      populated — a row with null provenance fails ingestion.
- [ ] FTS index built; embeddings populated for every chunk (model per ADR 0003
      decision); `make ingest-slice` is the single reproducible entrypoint.
- [ ] Acceptance: `SELECT count(*) FROM chunks` > 0 for every slice source_id;
      spot-check 5 chunks resolve back to a real page/section.

### M2 — Hybrid retrieval live (weeks 2-3)
- [ ] `/ask?debug=retrieval` returns fused (RRF, k=60) top-N with per-chunk
      provenance and both component ranks — retrieval-only, no generation.
- [ ] Sanity recall measured on the Chris-reviewed golden subset and recorded
      in `evals/reports/` (target: relevant chunk in top-10 for ≥80% of
      reviewed in-corpus questions; if missed, log — don't silently tune).
- [ ] Deterministic tests cover RRF fusion math and FTS/dense agreement on
      fixture chunks (no model calls).

### M3 — Reranker + citations (week 3)
- [ ] Reranker (ADR 0003 choice) behind `RERANK_ENABLED`, default OFF; A/B
      recall numbers (flag on vs off) recorded in `evals/reports/`.
- [ ] `/ask` returns composed answers where every evidence-backed claim carries
      a chunk-level citation `{source, document, section/page, chunk_id}` that
      RESOLVES (deterministic check: every cited chunk_id exists and its
      provenance matches the rendered citation).
- [ ] Response conforms to a published JSON schema (schema validity is a
      deterministic check).

### M4 — Abstention gate (week 4)
- [ ] Confidence gate (mechanism per ADR 0003 decision) returns a structured
      "cannot answer from this corpus" refusal when fused evidence is weak.
- [ ] All 5 out-of-corpus golden questions refuse correctly (deterministic
      check); refusal is a documented product feature with its own response
      schema, not an error.
- [ ] Deterministic checks green in CI (the ci-analyst job).

### M5 — Evals, observability, cost (weeks 4-5)
- [ ] Golden set grown toward 100 reviewed Q/A pairs (≥50 minimum), all
      human-reviewed by Chris.
- [ ] RAGAS wired (`make eval-ragas`); `make eval-report` writes a committed
      report (RAGAS metrics + deterministic results + latency/cost bounds).
- [ ] OTel traces flowing to self-hosted Langfuse for every /ask.
- [ ] Spend cap enforced and TESTED: budget exhausted in a test → 429 + refusal
      body; `query_log` rows carry tokens/cost/latency/decision.
- [ ] Public metrics endpoint serves p50/p95 latency + cost per query from
      `query_log` (cache hit rate added only when caching lands — not before).

### M6 — Ship (weeks 5-6)
- [ ] Live URL deployed (host per ADR 0003 decision); /healthz green; metrics
      page public.
- [ ] README failure-modes section written (what it refuses, known weak spots,
      honest limitations).
- [ ] Eval report committed from the deployed configuration.
- [ ] Writeup #1 published; demo link sent to 10 named people
      (`docs/plan/WRITEUPS.md`). Writeups #2-#3 follow fortnightly.

## Engineering cap (standing)

No new test infrastructure beyond what the evals need. No speculative
abstractions. One CI job. When polish competes with shipping, shipping wins.

## Open decisions (ADR 0003)

Decided 2026-06-11 (adopted under Chris's delegation; see the ADR's decision
record): D1 reranker = Cohere Rerank 4 Fast · D2 embeddings =
text-embedding-3-small (1536) · D4 abstention = fused-RRF threshold +
min-hits. **Still open — D3 hosting (Chris: needs your account):** Fly.io /
Hetzner / Railway; on-box vs managed Postgres; Langfuse placement.
