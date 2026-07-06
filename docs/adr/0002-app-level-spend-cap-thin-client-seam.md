# ADR 0002 — App-level spend cap now, gateway later, one thin LLM client seam

- Status: **Accepted** (locked in the Hero #1 plan, 2026-06-11)
- Deciders: Chris (plan locked before this session)
- Scope: `projects/wealthlens-analyst/` (WealthLens v2)

## Context

The product promise is "cheap, reliable, provably valuable": visible cost per
query and a **hard** spend cap are product features, not ops afterthoughts.
No LLM gateway (LiteLLM or similar) exists in this repo yet — that is Hero #2.
The cap must therefore be enforced *inside the app* today, in a way that a
future gateway can replace without touching call sites.

## Decision

1. **Hard spend cap enforced in the app.**
   - A `budgets` table holds the cap (e.g. monthly GBP/USD limit) and a
     `query_log` table records per-request `tokens_in`, `tokens_out`, `cost`,
     `latency_ms`, and `decision` (answered / refused / over-budget / error).
   - FastAPI **middleware** checks accumulated spend before any model call and
     returns **429** with a structured refusal body when the cap is exceeded.
     Over-budget refusal is a *visible, tested* behaviour (deterministic eval
     check), not an error path.
   - The cap value comes from configuration (`BUDGET_MONTHLY_CAP_GBP`), defaulting
     to a conservative value; like all new features in this repo it ships
     **default-safe** (a missing/zero budget row blocks spend rather than
     allowing unlimited spend — fail-closed).
2. **One thin client seam.** Every model call (generation, embeddings if
   hosted, reranking if API-based) goes through a single module,
   `wealthlens_analyst/llm/client.py`, which defines an `LLMClient`
   Protocol/ABC (complete + embed + count/cost accounting) and one concrete
   provider implementation (Anthropic first; the seam makes the provider
   swappable). **No other module may import a provider SDK.** This is the
   contract that makes Hero #2 (a LiteLLM gateway) a config flip: the gateway
   becomes one more `LLMClient` implementation.
3. **Migrations via Alembic** (see ADR 0001 §5): `budgets` and `query_log` are
   the second pair of migration drafts alongside `chunks`/`embeddings`.

## Consequences

- Cost per query is computable from `query_log` alone, which is exactly what
  the public metrics page (p50/p95 latency, cost/query) serves — no separate
  metering system.
- The cap is enforceable and testable offline: deterministic checks can seed a
  budget row, exhaust it, and assert the 429 refusal shape without any model
  call.
- The seam costs one indirection per call and forbids "quick" direct SDK use
  anywhere else; that discipline is the price of the later gateway flip.
- Token/cost accounting lives behind the seam too, so a provider change cannot
  silently break the metrics page.

## Alternatives considered (rejected)

- **Adopt LiteLLM gateway now:** a second deployed service + config surface
  before a single query is served; violates the engineering cap and the locked
  sequencing (gateway is Hero #2).
- **Soft cap (alerting only):** a demo with a public URL and no hard stop is a
  credit-card incident waiting to happen; the cap being *hard* is part of the
  portfolio story.
- **Per-provider call sites with shared helper:** drifts immediately; the
  Protocol + single-module rule is enforceable by review and by a trivial
  grep/CI check.
