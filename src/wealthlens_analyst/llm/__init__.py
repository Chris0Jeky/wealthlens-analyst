"""The LLM client seam (ADR 0002).

Every model call in this service — generation, hosted embeddings, API
reranking — goes through client.py. No other module may import a provider
SDK. This is the contract that makes a future LiteLLM gateway (Hero #2) a
config flip instead of a refactor.
"""
