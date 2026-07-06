# ADR 0001 — Hybrid retrieval: Postgres FTS + pgvector, fused with RRF; reranker behind a flag

- Status: **Accepted** (locked in the Hero #1 plan, 2026-06-11)
- Deciders: Chris (plan locked before this session)
- Scope: `projects/wealthlens-analyst/` (WealthLens v2, evidence-backed research analyst)

> Note on location: workspace decision records live in the private HQ repo
> (`../hq-private/projects/wealthlens/memories/decisions/`, date-named; moved
> there 2026-06-13). Product ADRs live here in `docs/adr/` (numbered) instead,
> because they must travel with the product when it is extracted to its own repo.

## Context

WealthLens v2 answers questions about official UK wealth statistics with
chunk-level citations and honest abstention. Retrieval quality drives everything
downstream (citations, abstention confidence, eval scores). The corpus is small
(a frozen slice: ONS WAS, HMRC distributional statistics, 3-5 IFS/Resolution
Foundation reports) and the query volume is low. The repo already has Postgres
in its declared stack and no existing search infrastructure.

## Decision

1. **Hybrid retrieval.** Every query runs two retrievers over the same `chunks`
   table:
   - **Lexical:** Postgres full-text search (`tsvector`/`tsquery`, GIN index).
   - **Dense:** pgvector cosine similarity over an `embeddings` column/table.
2. **Fusion: Reciprocal Rank Fusion (RRF).** `score(d) = Σ 1/(k + rank_i(d))`
   with the standard `k = 60`. RRF is rank-based, so it needs no score
   calibration between FTS and cosine similarity — the property that makes it
   the safe default for hybrid fusion.
3. **Reranker behind a feature flag** (`RERANK_ENABLED`, default **OFF** per the
   repo's config-defaults rule). The fused top-N (default 20) is optionally
   reranked to top-k before answer composition. The reranker choice (Cohere
   Rerank API vs self-hosted BGE) was settled by **ADR 0003** (decided
   2026-06-11: Cohere Rerank 4 Fast — see its decision record).
4. **Provenance is captured at ingestion time.** Each chunk row carries
   `source_id` (FK to the registry id in `registries/sources.yml`),
   `document_id`, `section`, `page`, and `span`. Citations resolve from these
   columns; nothing is reconstructed at query time.
5. **One database, one migration tool.** Postgres (pgvector extension) with
   **Alembic** migrations. The repo previously had *no* database and *no*
   migration tool (the dashboard backend serves static CSVs), so Alembic is
   adopted here as the de facto SQLAlchemy/FastAPI standard rather than
   "reusing" a tool that did not exist. It is scoped to the analyst subtree
   (`projects/wealthlens-analyst/migrations/`).

## Consequences

- Both retrievers share one source of truth (the `chunks` table), so ingestion
  writes once and provenance cannot drift between lexical and dense paths.
- RRF keeps the fusion logic deterministic and testable without model calls —
  the eval harness's deterministic checks can exercise it directly.
- The flag means the system must be *useful without the reranker*; the reranker
  is an upgrade, not a dependency. M2 ships flag-off; M3 turns it on.
- Abstention (the confidence gate, see the locked plan) reads the *fused*
  evidence strength; the signal is fixed by ADR 0003 D4 (fused-RRF threshold + min-hits).
- Alembic adds one new dev dependency and a `migrations/` directory to the
  analyst subtree only; no other part of the monorepo gains a database.

## Alternatives considered (rejected)

- **Dedicated vector DB (Qdrant/Weaviate/etc.):** a second stateful service for
  a corpus of a few thousand chunks; violates the engineering cap and the
  hosting budget. Postgres already has to exist for budgets/query logs.
- **Dense-only retrieval:** official statistics are full of exact terms
  ("Gini", "decile", "IHT nil-rate band", years, table numbers) where lexical
  match is load-bearing; dropping FTS measurably hurts that class of query.
- **Score-weighted fusion (e.g. normalised linear combination):** requires
  calibrating incomparable score distributions; RRF avoids the problem.
