# ADR 0003 — Reranker, embedding model, hosting, and abstention mechanism (open-decisions memo)

- Status: **Accepted** — D1/D2/D4 adopted 2026-06-11 under Chris's explicit
  blanket delegation ("make all the reasonable decisions that you can make on
  your own"); D3 (hosting) decided by Chris 2026-06-13 (**Hetzner CAX21**, per
  the memo's recommendation). Each followed the memo; Chris can override any
  before the affected milestone ships. Provisioning is H1-30, once the Hetzner
  account exists.
- Date: 2026-06-11. All prices/versions below were web-verified on this date
  against official pages (source URLs inline). Numbers marked *unverified*
  could not be confirmed on an official page and must not be relied on.
- Context: low query volume, ~thousands of chunks, hard spend cap, solo
  maintainer, public always-on demo URL, target well under £20/month all-in.

## D1 — Reranker: Cohere Rerank API vs self-hosted BGE

| | Cohere Rerank API | Self-hosted BGE (`BAAI/bge-reranker-v2-m3`) |
|---|---|---|
| Model | `rerank-v4.0-fast` / `rerank-v4.0-pro` — [docs.cohere.com/docs/rerank](https://docs.cohere.com/docs/rerank) | 0.6B cross-encoder, Apache 2.0 — [model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Cost / 1k queries | **$2.00** (Fast) / $2.50 (Pro) — [cohere.com/pricing](https://cohere.com/pricing); 1 search unit = 1 query + up to 100 docs | $0 marginal (compute only) |
| Added p95 latency | Not published (Fast is "lowest latency", qualitative only) — *unverified* | CPU at low QPS: order of seconds for 20-50 candidates (*estimate, no published benchmark*) |
| Ops burden | None (API) | One more service: ~1.2GB weights at fp16 (~2.4GB fp32; *estimates from 0.6B params, not benchmarked*), serving process, version pinning |
| Dev / free tier | Trial keys: free, 10 req/min, 1,000 calls/mo, non-production only — [docs.cohere.com/docs/rate-limits](https://docs.cohere.com/docs/rate-limits) | n/a |

**Recommendation:** Cohere Rerank 4 Fast. At this volume it's ~$2/month and
zero-ops; the trial key covers all development. Keep bge-reranker-v2-m3 as
the documented exit ramp behind the same `rerank()` seam (rerank-v3.5's price
no longer appears on [cohere.com/pricing](https://cohere.com/pricing) as of
2026-06-11, though third-party listings still show $2/1k — pricing churn is
itself a risk signal).

## D2 — Embedding model: one hosted + one open-weights

| | Hosted: OpenAI `text-embedding-3-small` | Open-weights: `BAAI/bge-m3` |
|---|---|---|
| Price / 1M tokens | **$0.02** — [official model page](https://developers.openai.com/api/docs/models/text-embedding-3-small) | $0 (one-off CPU batch job is fine at corpus scale) |
| Dims (pgvector) | 1536 → ~6.0KB/vector, ~62MB per 10k chunks | 1024 (MIT licence, 8192-token window) → ~4.0KB/vector, ~41MB per 10k chunks |
| Index | Standard HNSW fine (≤2,000-dim limit verified current — [pgvector README](https://github.com/pgvector/pgvector)) | Standard HNSW fine |

Also priced: `text-embedding-3-large` $0.13/1M, 3072 dims
([model page](https://developers.openai.com/api/docs/models/text-embedding-3-large))
— **exceeds the 2,000-dim HNSW limit** at full width (needs halfvec or
dimension-shortening); not worth it at this scale. Cohere Embed 4: $0.12/1M
([cohere.com/pricing](https://cohere.com/pricing)). Voyage `voyage-4-lite`
$0.02/1M + 200M free tokens
([voyage pricing](https://docs.voyageai.com/docs/pricing)) — a third vendor
for negligible savings.

**Recommendation:** `text-embedding-3-small` to ship (embedding the whole
corpus costs well under $0.10; one vendor key already needed if Cohere rerank
is chosen — note OpenAI embeddings add a second provider key either way).
bge-m3 is the open-weights fallback if a zero-external-dependency story
matters more. Caveat: hosted embeddings put a second provider behind the
`llm/client.py` seam — the seam supports it (ADR 0002), the cap meters it.

## D3 — Hosting

| Option | £/month (approx) | Postgres | pgvector | Ops (1-5) | Caveat |
|---|---|---|---|---|---|
| **Hetzner CAX11** (Arm 2vCPU/4GB) + Docker Compose, DB on-box | **~£4.2** (€4.49 + €0.50 IPv4 — [hetzner price adjustment, Apr 2026](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)) | on-box (`pgvector/pgvector` image) | yes | 4 | You own patching, TLS (Caddy), backups (nightly `pg_dump` cron) |
| Fly.io machine + Neon Free | ~£2-4 (Fly ~$2-4 — [fly.io pricing](https://fly.io/docs/about/pricing/); Neon $0 — [neon.com/pricing](https://neon.com/pricing)) | Neon managed | yes (all plans) | 2 | Neon Free autosuspends after 5 min idle (auto-wakes; brief cold start). Fly has **no free tier** for new orgs; Fly *Managed* Postgres starts $38/mo — out of budget |
| Railway Hobby | ~£4-11 ($5 incl. credit + metered — [railway pricing](https://docs.railway.com/reference/pricing/plans)) | pgvector template (standard image lacks it) | yes via template | 2 | Always-on app+DB likely exceeds the $5 credit; cost is usage-metered |
| Supabase Free (DB only) | £0 — [supabase.com/pricing](https://supabase.com/pricing) | Supabase managed | yes — [pgvector docs](https://supabase.com/docs/guides/database/extensions/pgvector) | 1 | **Free projects pause after 1 week idle, manual restore** — a public demo link that silently dies. Pro ($25/mo) eats the budget |

**Langfuse sizing note (affects D3):** self-hosted Langfuse v3+ requires
Postgres + ClickHouse + Redis/Valkey + S3-compatible storage + web/worker
containers ([langfuse.com/self-hosting](https://langfuse.com/self-hosting)) —
a real memory footprint. On Hetzner that argues for stepping up to an 8GB
machine (CAX21-class, roughly €2-4/mo more — *approximate; confirm against
[Hetzner's current price list](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)
when provisioning*) or running Langfuse on a second small box; worth deciding
together with D3. (The locked decision says self-hosted; this is a sizing
consequence, not a re-litigation.)

**Recommendation:** Hetzner CAX11 (or CAX21 if Langfuse shares the box),
Docker Compose, everything on-box. Cheapest genuinely-always-on option by a
wide margin, no autosuspend surprises on a public URL, and the ops burden
(unattended-upgrades, Caddy, pg_dump cron) is squarely within a DevSecOps
background. Fallback: Fly + Neon Free if zero-ops matters more than the
cold-start blink.

## D4 — Abstention mechanism

Options, cheapest-testable first:
1. **Threshold on the fused RRF score** (+ a min-hits guard, e.g. "top fused
   score < τ OR fewer than m chunks above τ′ → refuse"). Zero model calls,
   fully deterministic, testable in CI with fixture chunks. RRF scores are
   bounded (≤ Σ 1/(k+1)) and rank-derived, so τ is stable across corpora.
2. Threshold on the reranker relevance score — better calibrated than RRF,
   still deterministic, but couples abstention to the rerank flag being ON
   (and to D1's vendor) — the gate must also work flag-off.
3. Small judge call ("does this evidence answer the question?") — most
   flexible, but costs money per refusal, adds latency, and is NOT
   deterministically testable; it would weaken the eval story.

**Recommendation:** Option 1 (fused-score threshold + min-hits), calibrated
once against the reviewed golden set (in-corpus vs out-of-corpus separation),
with the gate signal logged to `query_log.gate_signal` for auditability.
Revisit only if M4's refusal checks can't reach green on thresholds alone.

## Dependency versions (verified against PyPI, 2026-06-11)

Pinned as floors in `projects/wealthlens-analyst/pyproject.toml`:
fastapi 0.136.3 · uvicorn 0.49.0 · pydantic 2.13.4 · SQLAlchemy 2.0.50 ·
alembic 1.18.4 · pgvector (py) 0.4.2 · psycopg 3.3.4 · httpx 0.28.1 ·
anthropic 0.109.1 · jsonschema 4.26.0 · ragas 0.4.3 · langfuse 4.7.1 ·
opentelemetry-api/sdk 1.42.1 · opentelemetry-instrumentation-fastapi 0.63b1
(beta-only versioning) · pytest 9.0.3.
Postgres pgvector extension: v0.8.2 ([tags](https://github.com/pgvector/pgvector/tags)).
Convention notes: mypy 2.1.0 is current but the monorepo caps `<2` — the
analyst matches the repo (`mypy>=1.10,<2`). Langfuse SDK is now v4
(OTel-based since v3); self-hosted platform requirements above.

## Decision record (D1/D2/D4 recorded under delegation; D3 for Chris)

- [x] D1 reranker: **Cohere Rerank 4 Fast** (per recommendation; adopted by
      delegation 2026-06-11; bge-reranker-v2-m3 documented as exit ramp)
- [x] D2 embedding model: **OpenAI text-embedding-3-small, 1536 dims** (per
      recommendation; adopted by delegation 2026-06-11; encoded in migration
      0002_embeddings)
- [x] D3 hosting (+ Langfuse placement): **Hetzner CAX21** (8GB Arm, ~£7/mo,
      Docker Compose, everything on-box incl. self-hosted Langfuse) — chosen by
      Chris 2026-06-13. Next: Chris creates the Hetzner account, then the agent
      provisions (H1-30: Docker Compose, Caddy TLS, nightly pg_dump cron, Alembic + ingest).
- [x] D4 abstention mechanism: **fused-RRF-score threshold + min-hits guard**
      (per recommendation; adopted by delegation 2026-06-11; calibrated against
      the reviewed golden set in H1-21)
