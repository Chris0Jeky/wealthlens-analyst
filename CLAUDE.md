# CLAUDE.md — WealthLens Analyst

Citation-first RAG service over a **frozen 8-source slice** of official UK wealth statistics
(ONS WAS, HMRC distributional, IFS/RF reports). `POST /ask` runs Postgres FTS ∥ pgvector dense
→ RRF fusion (k=60) → abstention gate → cited generation → citation resolution, and serves an
answer **only if every citation resolves**; otherwise an honest structured refusal. Every
request that *reaches the handler* writes one `query_log` accounting row (tokens, cost,
latency, decision) — a request rejected by FastAPI validation (422: malformed question,
unknown `debug` value) never reaches it and logs nothing, so `query_log` is not a record of
all inbound traffic.
Python 3.11 · FastAPI · SQLAlchemy Core · Alembic · pytest · ruff + **strict mypy**.

Global laws (review, merge, tiers, worktrees) live in `~/.claude/CLAUDE.md` and are injected
automatically — not restated here. Authority: `.agent-harness/tier.json` (**T2**, push free,
merge free). Human-blocked items: `HUMAN_TODO.md`.

## Run it

```bash
make db-up                      # pgvector/pgvector:pg17 on localhost:15432 (docker-compose.yml)
make install                    # pip install -e ".[dev,evals]"
cp .env.example .env            # then FILL IN or CLEAR OPENAI_API_KEY — the copied placeholder
                                # (`your-openai-key`) is NON-EMPTY, and ingestion gates
                                # embedding on truthiness, so leaving it fails on auth instead
                                # of taking the keyless FTS-only path.
                                # Then export: set -a; . ./.env; set +a  (config.py reads the
                                # PROCESS env — there is NO dotenv auto-load)
alembic upgrade head            # hand-written revisions in migrations/versions/
make ingest-slice               # chunk → provenance gate → write; embeds iff OPENAI_API_KEY set
make dev                        # uvicorn --factory ... 127.0.0.1:8100
```

**`make ingest-slice` is not yet runnable from a fresh clone.** It reads the tabular CSVs
from `data/processed/`, which is gitignored and empty on checkout — they are the upstream
wealthlens-hq dashboard pipeline's outputs. Without them `ingest_slice()` fails loudly with
`no processed source CSVs found in …`. Generate or copy them in first; making acquisition
self-contained is H1-34, still open.

No DB and no API key are needed for the test suite or CI — every test stubs the engine and the
LLM seam. Windows: `make` and `docker` are on PATH here; if `make` is missing, run the
underlying `python -m ...` lines from the Makefile directly.

## Proving checks — pick the narrowest that exercises your change

All timings measured 2026-07-27 in a clean worktree (`python -m pytest`, repo `.venv`).

| Change touches | Command | Measured |
|---|---|---|
| `retrieval/` | `pytest tests/test_fts.py tests/test_dense.py tests/test_fuse_rrf.py -q` | 24 passed, 0.2s |
| `answer/` | `pytest tests/test_abstain.py tests/test_citations.py tests/test_compose.py -q` | 50 passed, 0.3s |
| `api/` | `pytest tests/test_api.py tests/test_schemas.py -q` | 48 passed, 0.8s |
| `llm/client.py` | `pytest tests/test_llm_client.py -q` | 16 passed, 0.03s |
| `budget/models.py` | `pytest tests/test_budget_models.py -q` | 12 passed, 0.2s |
| `budget/middleware.py` | **NOT covered** — nothing imports it but `tests/test_imports.py`, and that only imports the module; `budget_guard` is never called. Write the test with H1-27 | n/a |
| `ingest/` | `pytest tests/test_ingest_write.py tests/test_slice_corpus.py -q` | 44 passed, 0.3s |
| `scripts/fetch_corpus.py` | **NOT covered** — no test imports or runs it. Nearest real check needs network + a populated `data/raw/`: `python scripts/fetch_corpus.py --verify` | not run here |
| `evals/golden/`, `evals/checks/deterministic.py` | `python evals/checks/deterministic.py` | `20 records (0 reviewed, 20 draft, 5 refusal probes) · OK` |
| `evals/checks/check_citations_live.py`, `check_compose_live.py` | **not reached by `deterministic.py`** (it imports neither). Each needs analyst-db ingested + a real `OPENAI_API_KEY` and **spends** (~1 embed + 1 generation): `python evals/checks/check_<name>_live.py` | not run here |
| new/moved module, stub filled | `pytest tests/test_imports.py tests/test_golden_draft_guard.py -q` | 6 passed, 0.6s |
| `migrations/` | **no unit coverage** — needs a live DB: `make db-up && alembic upgrade head` | not run here |
| anything non-trivial | `make lint && make test && python evals/checks/deterministic.py` | 200 passed; ruff+format clean; mypy 24 files, no issues |

That last row **is** CI (`.github/workflows/ci.yml`, one job by design), and all of it runs in
well under a minute locally. RAGAS is deliberately not in CI — it spends real money.

## Repo map

```
src/wealthlens_analyst/
  config.py    env → frozen Settings; malformed BUDGET_MONTHLY_CAP_GBP fails LOUDLY at startup
  db.py        engine factory + chunks Core table
  retrieval/   fts.py dense.py fuse_rrf.py rerank.py(H1-16 stub)      ADR 0001
  answer/      compose.py citations.py abstain.py
  llm/client.py  THE provider seam — the only module that may import an SDK   ADR 0002
  budget/      models.py middleware.py (H1-27 stub)                    ADR 0002
  api/         app.py routes.py schemas.py  → /ask /healthz /metrics/data(H1-29 stub)
  ingest/      slice_corpus.py fetch_documents.py
migrations/ evals/{golden,checks,run_ragas.py} registries/sources.yml data/corpus_manifest.yml
docs/plan/HERO1_PLAN.md · docs/adr/0001-0003 · tasks/hero1-backlog.md · tests/
```

Status: **M0–M4 done, M5 in progress.** Stubs raising `NotImplementedError` are pinned by
`tests/test_imports.py` — gutting one to `return []` will not pass.

## Pitfalls specific to this repo

- **The plan is FINAL.** `docs/plan/HERO1_PLAN.md` (M0–M6) and `tasks/hero1-backlog.md` are
  locked. Do not re-plan, re-architect, or re-sequence. Open questions are ADR decisions and
  Chris's, not yours.
- **Never fabricate ground truth** — golden answers, statistics, citations. All 20 golden
  records are DRAFT and must stay answer-empty; `tests/test_golden_draft_guard.py` enforces it.
- **Never import a provider SDK outside `llm/client.py`.** Nothing enforces this automatically
  — check with `grep -rn "import openai\|import anthropic\|import cohere" src/` before pushing.
- **Never add a corpus source** before its live URL exists; corpus identity lives in
  `registries/sources.yml` + `data/corpus_manifest.yml` and is frozen until v1 ships.
- **Never skip the metering path** — every model call is accounted; a served outcome whose
  `query_log` row cannot be written must 500, not serve unmetered spend.
- Embedding dimension **1536 is baked into migration `0002_embeddings`**; pointing
  `EMBEDDING_MODEL` at a different width needs a new revision, not an env edit.
- Migrations are hand-written; `target_metadata = None`, so `--autogenerate` produces nothing.
- Secrets: real values live in the gitignored `.env` only; `.env.example` documents the names.
  This remote is **public** — never commit a key, never move private/planning material here.
- No new test infrastructure beyond what the evals need. One CI job. Shipping beats polish.
