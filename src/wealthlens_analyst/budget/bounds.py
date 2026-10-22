"""Conservative /ask cost bound, using the provider seam's existing prices.

Byte-level tokenizers cannot emit more ordinary tokens than UTF-8 bytes. We
reserve one token per byte, plus 1024 tokens for provider framing. Embedding
reserves four bytes per question character plus the same generous overhead.
Generation reserves MAX_PROMPT_BYTES (64 KiB) content bytes plus framing and compose's configured
maximum output (including reasoning). The route checks the EXACT composed
system/user content against that byte limit before generation; corpus text and
provenance are otherwise unbounded database TEXT, even with only eight hits.
Oversized evidence is refused intact rather than truncated into changed facts.

Bound = embedding_price(4 * len(question) + 1024)
      + generation_price(MAX_PROMPT_BYTES + 1024, compose._MAX_ANSWER_TOKENS), rounded UP
to query_log's GBP scale (8 decimal places). Unpriced models raise ValueError.
This bounds the same price/FX estimates used by query_log, not a provider invoice.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

from wealthlens_analyst.answer import compose
from wealthlens_analyst.config import Settings
from wealthlens_analyst.llm.client import embedding_cost_gbp, generation_cost_gbp
from wealthlens_analyst.retrieval.fts import ChunkHit

# Eight ~500-token evidence chunks plus provenance and the system prompt measure
# roughly 20-30 KB; 64 KiB leaves headroom so real answers are never refused here.
# Worst-case reservation at that size is a few pence against the monthly cap.
MAX_PROMPT_BYTES = 65_536
_FRAMING_TOKENS = 1024


def request_cost_bound_gbp(question: str, settings: Settings) -> Decimal:
    """Reserve both priced model calls, even for embedding-only debug requests."""
    embedding = embedding_cost_gbp(settings.embedding_model, 4 * len(question) + _FRAMING_TOKENS)
    generation = generation_cost_gbp(
        settings.analyst_model, MAX_PROMPT_BYTES + _FRAMING_TOKENS, compose._MAX_ANSWER_TOKENS
    )
    return (Decimal(str(embedding)) + Decimal(str(generation))).quantize(Decimal("0.00000001"), rounding=ROUND_CEILING)


def prompt_fits_bound(question: str, evidence: list[ChunkHit]) -> bool:
    """Check the content compose will send, including all provenance and UTF-8.

    Keep the user-message envelope aligned with compose_answer; a test compares
    this measurement against that function's actual client arguments.
    """
    prompt = f"Question: {question}\n\nEvidence:\n\n{compose._format_evidence(evidence)}"
    return len(compose._SYSTEM_PROMPT.encode("utf-8")) + len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES
