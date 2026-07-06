# Migrations (Alembic)

Hand-written revisions in `versions/` (no autogenerate: the schema is small
and each revision documents its rationale). Promoted from the kickoff drafts
by task H1-05.

The repo had **no database and no migration tool** before this product
(the dashboard backend serves static CSVs), so Alembic was adopted here —
decision recorded in ADR 0001 §5. Scope is this subtree only.

## Usage

```bash
docker compose up -d analyst-db        # from the repo root (pgvector/pgvector:pg17)
cd projects/wealthlens-analyst
alembic upgrade head                   # DATABASE_URL env overrides alembic.ini
alembic downgrade base                 # full reversal (extension is left in place)
```

## Revisions

1. `0001_chunks` — corpus chunks with ingestion-time provenance columns
   (source_id, document_id, section, page, span) + STORED tsvector + GIN.
2. `0002_embeddings` — pgvector extension + VECTOR(1536) (ADR 0003 D2:
   text-embedding-3-small) + HNSW cosine index.
3. `0003_budgets` — the hard spend cap (ADR 0002); one-active partial unique index.
4. `0004_query_log` — per-query tokens/cost/latency/decision; feeds the
   public metrics page. Stores question hashes, not raw text.
