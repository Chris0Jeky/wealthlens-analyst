"""WealthLens Analyst — evidence-backed research analyst over UK wealth statistics.

Citation-first hybrid retrieval (Postgres FTS + pgvector, RRF-fused) with
honest abstention, a committed eval harness, and a hard spend cap.

Locked architecture: docs/adr/0001 (retrieval) and docs/adr/0002 (spend cap +
LLM client seam) at the repo root. Do not re-architect.
"""

__version__ = "0.1.0.dev0"
