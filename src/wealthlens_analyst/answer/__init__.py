"""Answer composition: cited generation, citation resolution, abstention.

Every evidence-backed claim carries a chunk-level citation that resolves to
{source, document, section/page, chunk_id}. When fused retrieval evidence is
weak, the system returns a structured refusal — abstention is a product
feature, not an error path.
"""
