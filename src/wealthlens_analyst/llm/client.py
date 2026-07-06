"""The thin LLM client seam — the ONLY module that touches provider SDKs.

ADR 0002: one Protocol, concrete implementations behind it, token/cost
accounting included in every result so the budget middleware and the public
metrics page never depend on a provider's shape.

Provider SDK imports happen INSIDE methods (lazy), so importing this module —
and everything above it — never requires the provider package or credentials.

Provider: OpenAI for BOTH embeddings (ADR 0003 D2: text-embedding-3-small, 1536
dims) and generation (gpt-5.4-mini — Chris's provider call 2026-06-30, superseding
the claude-sonnet-4-6 default; not an ADR D-decision). Model ids come from
configuration (EMBEDDING_MODEL / ANALYST_MODEL), never hard-coded. Embeddings
(`embed`) land here in H1-11; generation (`complete`) in H1-18.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wealthlens_analyst.config import load_settings


@dataclass(frozen=True)
class CompletionResult:
    """A model completion plus the accounting the budget layer requires."""

    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_gbp: float
    #: Provider stop reason ("stop", "length", ...). Callers must treat
    #: "length" as a truncated answer, not a complete one — a cut-off figure
    #: is worse than no answer for a statistics product.
    finish_reason: str = ""


@dataclass(frozen=True)
class EmbeddingResult:
    """Embeddings plus accounting. Vectors are row-aligned with the input texts."""

    vectors: list[list[float]]
    model: str
    tokens_in: int
    cost_gbp: float


# --- Pricing -----------------------------------------------------------------
# OpenAI list prices in USD per 1M tokens (verified 2026-06: text-embedding-3-small
# is $0.02/1M, -3-large $0.13/1M). Cost is reported in GBP because the hard spend
# cap is GBP (BUDGET_MONTHLY_CAP_GBP, ADR 0002); convert at a documented approximate
# FX rate. Embedding the frozen corpus is sub-penny, so FX precision is immaterial —
# this is an internal cost ESTIMATE, not a published figure. An UNKNOWN model fails
# loud rather than under-reporting cost (silent under-reporting would defeat the cap).
_EMBEDDING_USD_PER_1M_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}
# Generation list prices, (input, output) USD per 1M tokens. Verified 2026-07-02
# at developers.openai.com/api/docs/pricing (gpt-5.4-mini: $0.75 in / $4.50 out).
# Output tokens INCLUDE any reasoning tokens the model spends — OpenAI bills
# them as completion tokens, so usage.completion_tokens is the billable count.
_GENERATION_USD_PER_1M_TOKENS = {
    "gpt-5.4-mini": (0.75, 4.50),
}
_USD_TO_GBP = 0.79  # approximate, for internal cost estimation only (not a published stat)


def embedding_cost_gbp(model: str, tokens_in: int) -> float:
    """GBP cost of ``tokens_in`` embedding tokens for ``model``.

    Fails loud on an unknown model: the spend cap (ADR 0002) is computed from these
    costs, so a silently-zero cost for an unpriced model would let spend run unbounded.
    """
    if model not in _EMBEDDING_USD_PER_1M_TOKENS:
        raise ValueError(
            f"no price configured for embedding model {model!r}; add it to "
            "_EMBEDDING_USD_PER_1M_TOKENS so cost is never silently under-reported"
        )
    usd = tokens_in / 1_000_000 * _EMBEDDING_USD_PER_1M_TOKENS[model]
    return usd * _USD_TO_GBP


def generation_cost_gbp(model: str, tokens_in: int, tokens_out: int) -> float:
    """GBP cost of one generation call. Fails loud on an unknown model.

    Same rationale as embedding_cost_gbp: the hard cap (ADR 0002) sums these
    costs, so a silently-zero cost for an unpriced model would let spend run
    unbounded.
    """
    if model not in _GENERATION_USD_PER_1M_TOKENS:
        raise ValueError(
            f"no price configured for generation model {model!r}; add it to "
            "_GENERATION_USD_PER_1M_TOKENS so cost is never silently under-reported"
        )
    price_in, price_out = _GENERATION_USD_PER_1M_TOKENS[model]
    usd = tokens_in / 1_000_000 * price_in + tokens_out / 1_000_000 * price_out
    return usd * _USD_TO_GBP


class LLMClient(Protocol):
    """What the rest of the service is allowed to know about model providers.

    Implementations MUST fill in real token counts and cost — the hard spend
    cap (budget/middleware.py) and the metrics page are computed from these.
    """

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
        """Run one generation call."""
        ...

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts with the configured embedding model."""
        ...


class OpenAIClient:
    """Concrete OpenAI implementation: embeddings (H1-11) + generation (H1-18).

    ``import openai`` is lazy (inside the methods) so importing this module needs
    neither the SDK installed nor a key set. Model ids come from configuration.
    """

    def __init__(self, *, api_key: str, embedding_model: str, analyst_model: str) -> None:
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._analyst_model = analyst_model

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
        """Run one chat completion with the configured analyst model.

        ``max_tokens`` maps to the API's ``max_completion_tokens`` (the
        parameter current OpenAI models require), which caps output INCLUDING
        any reasoning tokens the model spends — callers should budget for
        that. Cost is computed from the returned usage via generation_cost_gbp
        (fail-loud on an unpriced model), never estimated locally.
        """
        if not self._analyst_model:
            raise RuntimeError("ANALYST_MODEL is not configured (see .env.example); cannot run generation")
        # Pre-flight the price table BEFORE the network call: an unpriced
        # model must be a zero-spend loud failure, not money burned and then
        # found to be unaccountable (ADR 0002).
        generation_cost_gbp(self._analyst_model, 0, 0)
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._analyst_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        usage = response.usage
        if usage is None:
            # Without usage there is no honest cost figure; under-reporting
            # (e.g. assuming 0) would defeat the spend cap, so fail loud.
            raise RuntimeError("OpenAI response carried no usage block; cannot account the call (ADR 0002)")
        cost = generation_cost_gbp(self._analyst_model, usage.prompt_tokens, usage.completion_tokens)
        if not response.choices:
            # Possible under provider-side filtering. The spend is real, so
            # return an accountable empty result (callers treat empty text as
            # a loud failure that still carries the accounting) rather than
            # crash on choices[0] and lose the cost.
            return CompletionResult(
                text="",
                model=self._analyst_model,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                cost_gbp=cost,
                finish_reason="no_choices",
            )
        choice = response.choices[0]
        return CompletionResult(
            text=choice.message.content or "",
            model=self._analyst_model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            cost_gbp=cost,
            finish_reason=choice.finish_reason or "",
        )

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed ``texts`` with the configured embedding model (one batched call).

        Vectors are sorted back into input order by the API's per-item ``index``, so
        the result is row-aligned with ``texts`` regardless of response ordering.
        """
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self._embedding_model, input=texts)
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        tokens_in = response.usage.prompt_tokens
        return EmbeddingResult(
            vectors=vectors,
            model=self._embedding_model,
            tokens_in=tokens_in,
            cost_gbp=embedding_cost_gbp(self._embedding_model, tokens_in),
        )


def get_client() -> LLMClient:
    """Return the configured client — the single composition point for the seam.

    Fails loudly when the OpenAI key or embedding model is unset: a retrieval or
    ingest call with no provider configured is a setup error, not something to
    paper over with a silent default.
    """
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured (see .env.example); cannot reach the model provider")
    if not settings.embedding_model:
        raise RuntimeError("EMBEDDING_MODEL is not configured (see .env.example)")
    if settings.analyst_model:
        # A CONFIGURED-but-unpriced generation model should kill the app at
        # startup (the api lifespan calls get_client), not burn unaccountable
        # spend on the first /ask. Empty stays allowed: embedding-only flows
        # (ingest) do not need ANALYST_MODEL, and complete() fail-louds on it.
        generation_cost_gbp(settings.analyst_model, 0, 0)
    return OpenAIClient(
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
        analyst_model=settings.analyst_model,
    )
