"""Hybrid retrieval: Postgres FTS + pgvector dense search, fused with RRF.

Architecture is locked by ADR 0001. The reranker (rerank.py) sits behind the
RERANK_ENABLED feature flag, default OFF.
"""
