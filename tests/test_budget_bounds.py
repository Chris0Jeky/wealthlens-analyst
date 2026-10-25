"""Cost bounds and the exact provider prompt they protect; no model or DB calls."""

from dataclasses import replace
from decimal import Decimal

import pytest

from wealthlens_analyst.answer import compose
from wealthlens_analyst.budget.bounds import MAX_PROMPT_BYTES, prompt_fits_bound, request_cost_bound_gbp
from wealthlens_analyst.config import load_settings
from wealthlens_analyst.llm.client import CompletionResult, embedding_cost_gbp, generation_cost_gbp
from wealthlens_analyst.retrieval.fts import ChunkHit


@pytest.mark.parametrize("embedding_model", ["text-embedding-3-small", "text-embedding-3-large"])
@pytest.mark.parametrize("question", ["q", "💷" * 2000])
def test_bound_covers_both_calls_and_rounds_up(embedding_model: str, question: str) -> None:
    settings = replace(load_settings(), embedding_model=embedding_model, analyst_model="gpt-5.4-mini")
    cost = Decimal(str(embedding_cost_gbp(embedding_model, 4 * len(question) + 1024))) + Decimal(
        str(generation_cost_gbp(settings.analyst_model, MAX_PROMPT_BYTES + 1024, compose._MAX_ANSWER_TOKENS))
    )
    bound = request_cost_bound_gbp(question, settings)
    assert cost <= bound < cost + Decimal("0.00000001")
    assert bound.as_tuple().exponent == -8


def test_bound_uses_composes_current_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = replace(load_settings(), embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")
    original = request_cost_bound_gbp("q", settings)
    monkeypatch.setattr(compose, "_MAX_ANSWER_TOKENS", compose._MAX_ANSWER_TOKENS + 1000)
    assert request_cost_bound_gbp("q", settings) - original == Decimal("0.00355500")


@pytest.mark.parametrize("oversized_field", ["text", "source_id", "document_id", "section"])
def test_prompt_check_matches_actual_compose_messages(monkeypatch: pytest.MonkeyPatch, oversized_field: str) -> None:
    question = "What is wealth? 💷"
    evidence = [ChunkHit(i, "source", "document", "section", None, None, "evidence 💷", i, 1.0) for i in range(1, 9)]
    sent_bytes: list[int] = []

    class FakeClient:
        def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
            sent_bytes.append(len(system.encode("utf-8")) + len(prompt.encode("utf-8")))
            return CompletionResult("Answer [chunk:1]", "fake", 1, 1, 0.0, "stop")

    monkeypatch.setattr(compose, "get_client", FakeClient)
    compose.compose_answer(question, evidence)
    assert sent_bytes[-1] < MAX_PROMPT_BYTES
    assert prompt_fits_bound(question, evidence)
    # Pad to the exact byte ceiling, then exceed it by one multibyte character.
    spare = MAX_PROMPT_BYTES - sent_bytes[-1]
    value = getattr(evidence[0], oversized_field)
    evidence[0] = replace(evidence[0], **{oversized_field: value + "x" * spare})
    compose.compose_answer(question, evidence)
    assert sent_bytes[-1] == MAX_PROMPT_BYTES
    assert prompt_fits_bound(question, evidence)
    evidence[0] = replace(evidence[0], **{oversized_field: getattr(evidence[0], oversized_field) + "💷"})
    compose.compose_answer(question, evidence)
    assert sent_bytes[-1] == MAX_PROMPT_BYTES + 4
    assert not prompt_fits_bound(question, evidence)
