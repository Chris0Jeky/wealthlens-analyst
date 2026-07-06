# WealthLens Analyst

Evidence-backed research analyst over official UK wealth statistics
(ONS Wealth and Assets Survey, HMRC distributional statistics, selected
IFS/Resolution Foundation reports). Citation-first hybrid RAG with honest
abstention, a committed eval harness, and a hard spend cap.

**Status:** M3 in progress — plain `POST /ask` answers: hybrid retrieval
(Postgres FTS + pgvector dense, RRF-fused) → cited generation (gpt-5.4-mini
through the client seam) → citation resolution, returning the published
response schema (`api/schemas.py`: `answer` with resolved citations + full
provenance, or an honest `refusal`; the `over_budget` 429 variant lands with
the cap, H1-27). Serving policy: an answer is returned only when EVERY citation
resolves — if any cited id is fabricated/unverifiable the request is refused
rather than served (a citation-first product never serves an uncited claim),
and any drift / out-of-range citation-shaped text is stripped from served
answers. `POST /ask?debug=retrieval` still returns the fused candidate list
(component ranks, no generation). Every request writes a query_log accounting
row (embed + generation spend). The confidence gate over weak evidence
(H1-21/H1-22), reranking (H1-16, needs a Cohere key) and PDF sources (H1-08)
are pending. Plan: `docs/plan/HERO1_PLAN.md`
(repo root) · backlog: `tasks/hero1-backlog.md` · decisions: `docs/adr/0001-0003`.

## Layout

- `src/wealthlens_analyst/` — the service (retrieval, answer, llm seam,
  budget, api, ingest). See `CLAUDE.md` for the subtree map and locked rules.
- `migrations/` — Alembic; hand-written revisions in `migrations/versions/`
  (see `migrations/README.md` for usage).
- `evals/` — golden set (human-reviewed; DRAFT questions await review),
  deterministic checks, RAGAS runner, committed reports.
- `tests/` — pytest (mypy strict, monorepo ruff rules).

## Quick start

```bash
pip install -e "projects/wealthlens-analyst[dev,evals]"
docker compose up -d analyst-db   # Postgres+pgvector on :15432 (repo root)
make ingest-slice     # chunk + write + embed the frozen corpus slice
make dev              # uvicorn on 127.0.0.1:8100
# Plain mode — a cited answer or an honest refusal (spends one generation):
curl -X POST "http://127.0.0.1:8100/ask" \
  -H "Content-Type: application/json" -d '{"question": "who holds the most wealth?"}'
# Retrieval diagnostics only (no generation):
curl -X POST "http://127.0.0.1:8100/ask?debug=retrieval" \
  -H "Content-Type: application/json" -d '{"question": "who holds the most wealth?"}'
make eval-golden-validate
```

Configuration: copy `.env.example` to `.env`. Never commit secrets.

## Failure modes

_To be written at M6 (see HERO1_PLAN). This section will document what the
system refuses to answer and why, known weak spots, and honest limitations._
