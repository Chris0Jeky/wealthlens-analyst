"""Unit tests for the LLM client seam (H1-11 embeddings).

DB-free and SDK-free: the OpenAI SDK is faked via sys.modules so the embed
mapping, row-alignment, and cost accounting are pinned without a network call or
the package installed. The live embedding call is verified against the local DB
during ingestion (like the SQL in retrieval/fts.py).
"""

from __future__ import annotations

import sys
import types

import pytest

from wealthlens_analyst.llm.client import (
    EmbeddingResult,
    OpenAIClient,
    embedding_cost_gbp,
    get_client,
)


def test_embedding_cost_known_model() -> None:
    # text-embedding-3-small = $0.02 / 1M tokens, converted at the documented FX.
    assert embedding_cost_gbp("text-embedding-3-small", 1_000_000) == pytest.approx(0.02 * 0.79)
    assert embedding_cost_gbp("text-embedding-3-small", 0) == 0.0


def test_embedding_cost_unknown_model_fails_loud() -> None:
    # An unpriced model must raise, never silently return 0 (the cap depends on cost).
    with pytest.raises(ValueError, match="no price configured"):
        embedding_cost_gbp("text-embedding-9-imaginary", 100)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, *, prompt_tokens: int) -> dict[str, object]:
    """Install a fake `openai` module; return a dict capturing the call args."""
    captured: dict[str, object] = {}

    class _Item:
        def __init__(self, embedding: list[float], index: int) -> None:
            self.embedding = embedding
            self.index = index

    class _Usage:
        def __init__(self, tokens: int) -> None:
            self.prompt_tokens = tokens

    class _Response:
        def __init__(self) -> None:
            # Deliberately OUT OF ORDER so the index-sort is exercised.
            self.data = [_Item([0.2] * 1536, 1), _Item([0.1] * 1536, 0)]
            self.usage = _Usage(prompt_tokens)

    class _Embeddings:
        def create(self, *, model: str, input: list[str]) -> _Response:
            captured["model"] = model
            captured["input"] = input
            return _Response()

    class _OpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.embeddings = _Embeddings()

    fake = types.ModuleType("openai")
    fake.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    return captured


def test_embed_maps_and_aligns_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_openai(monkeypatch, prompt_tokens=7)
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")

    result = client.embed(["alpha", "beta"])

    assert isinstance(result, EmbeddingResult)
    # Sorted back into input order by item.index: index 0 ([0.1...]) first, index 1 ([0.2...]) second.
    assert result.vectors[0][0] == pytest.approx(0.1)
    assert result.vectors[1][0] == pytest.approx(0.2)
    assert len(result.vectors) == 2
    assert len(result.vectors[0]) == 1536
    assert result.model == "text-embedding-3-small"
    assert result.tokens_in == 7
    assert result.cost_gbp == pytest.approx(7 / 1_000_000 * 0.02 * 0.79)
    # The configured model + key + batched input reached the SDK.
    assert captured["model"] == "text-embedding-3-small"
    assert captured["api_key"] == "sk-x"
    assert captured["input"] == ["alpha", "beta"]


def test_get_client_builds_openai_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("ANALYST_MODEL", "gpt-5.4-mini")
    client = get_client()
    assert isinstance(client, OpenAIClient)


def test_get_client_without_key_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_client()


def test_get_client_without_embedding_model_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL"):
        get_client()


def test_generation_cost_known_model() -> None:
    from wealthlens_analyst.llm.client import generation_cost_gbp

    # gpt-5.4-mini = $0.75/1M in + $4.50/1M out (verified 2026-07-02,
    # developers.openai.com/api/docs/pricing), converted at the documented FX.
    assert generation_cost_gbp("gpt-5.4-mini", 1_000_000, 0) == pytest.approx(0.75 * 0.79)
    assert generation_cost_gbp("gpt-5.4-mini", 0, 1_000_000) == pytest.approx(4.50 * 0.79)
    assert generation_cost_gbp("gpt-5.4-mini", 0, 0) == 0.0


def test_generation_cost_unknown_model_fails_loud() -> None:
    from wealthlens_analyst.llm.client import generation_cost_gbp

    with pytest.raises(ValueError, match="no price configured"):
        generation_cost_gbp("gpt-99-imaginary", 100, 100)


def _install_fake_openai_chat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    usage: bool = True,
    empty_choices: bool = False,
) -> dict[str, object]:
    """Install a fake `openai` module for chat completions; capture call args."""
    captured: dict[str, object] = {}

    class _Message:
        def __init__(self) -> None:
            self.content = text

    class _Choice:
        def __init__(self) -> None:
            self.message = _Message()
            self.finish_reason = "stop"

    class _Usage:
        def __init__(self) -> None:
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Response:
        def __init__(self) -> None:
            self.choices = [] if empty_choices else [_Choice()]
            self.usage = _Usage() if usage else None

    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            captured.update(kwargs)
            return _Response()

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _OpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.chat = _Chat()

    fake = types.ModuleType("openai")
    fake.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    return captured


def test_complete_maps_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from wealthlens_analyst.llm.client import CompletionResult, generation_cost_gbp

    captured = _install_fake_openai_chat(
        monkeypatch, text="answer [chunk:1]", prompt_tokens=1200, completion_tokens=340
    )
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")

    result = client.complete(system="sys rules", prompt="user question", max_tokens=900)

    assert isinstance(result, CompletionResult)
    assert result.text == "answer [chunk:1]"
    assert result.model == "gpt-5.4-mini"
    assert result.tokens_in == 1200
    assert result.tokens_out == 340
    assert result.cost_gbp == pytest.approx(generation_cost_gbp("gpt-5.4-mini", 1200, 340))
    # The seam passes the configured model, both messages, and maps max_tokens
    # to max_completion_tokens (what current OpenAI models require).
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["max_completion_tokens"] == 900
    messages = captured["messages"]
    assert messages == [
        {"role": "system", "content": "sys rules"},
        {"role": "user", "content": "user question"},
    ]


def test_complete_none_content_becomes_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK types content as optional; the seam must not propagate None.
    _install_fake_openai_chat(monkeypatch, text=None, prompt_tokens=10, completion_tokens=900)
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")
    assert client.complete(system="s", prompt="p", max_tokens=900).text == ""


def test_complete_without_usage_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    # No usage block -> no honest cost figure -> never silently assume 0 (ADR 0002).
    _install_fake_openai_chat(monkeypatch, text="x", prompt_tokens=0, completion_tokens=0, usage=False)
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")
    with pytest.raises(RuntimeError, match="usage"):
        client.complete(system="s", prompt="p", max_tokens=10)


def test_complete_without_analyst_model_fails_loud() -> None:
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="")
    with pytest.raises(RuntimeError, match="ANALYST_MODEL"):
        client.complete(system="s", prompt="p", max_tokens=10)


def test_complete_unpriced_model_fails_loud_before_any_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    # The pre-flight must fire BEFORE the network call: a misconfigured model
    # is a zero-spend loud failure, never money burned then unaccountable.
    captured = _install_fake_openai_chat(monkeypatch, text="x", prompt_tokens=1, completion_tokens=1)
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-99-imaginary")
    with pytest.raises(ValueError, match="no price configured"):
        client.complete(system="s", prompt="p", max_tokens=10)
    # The fake SDK captures every create() call; none happened.
    assert "model" not in captured


def test_complete_surfaces_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai_chat(monkeypatch, text="short", prompt_tokens=10, completion_tokens=5)
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")
    assert client.complete(system="s", prompt="p", max_tokens=10).finish_reason == "stop"


def test_get_client_rejects_a_configured_but_unpriced_analyst_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # A deployment with a typo'd ANALYST_MODEL must die at startup (the api
    # lifespan calls get_client), not on the first paid /ask.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("ANALYST_MODEL", "gpt-99-imaginary")
    with pytest.raises(ValueError, match="no price configured"):
        get_client()
    # Empty stays allowed: embedding-only flows (ingest) need no ANALYST_MODEL.
    monkeypatch.setenv("ANALYST_MODEL", "")
    assert get_client() is not None


def test_complete_empty_choices_returns_accountable_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provider-side filtering can return zero choices AFTER spending; the
    # seam must return an accountable empty result, not crash on choices[0]
    # and lose the cost.
    from wealthlens_analyst.llm.client import generation_cost_gbp

    _install_fake_openai_chat(monkeypatch, text="x", prompt_tokens=50, completion_tokens=0, empty_choices=True)
    client = OpenAIClient(api_key="sk-x", embedding_model="text-embedding-3-small", analyst_model="gpt-5.4-mini")
    result = client.complete(system="s", prompt="p", max_tokens=10)
    assert result.text == ""
    assert result.finish_reason == "no_choices"
    assert result.tokens_in == 50
    assert result.cost_gbp == pytest.approx(generation_cost_gbp("gpt-5.4-mini", 50, 0))
