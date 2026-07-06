"""Unit tests for cited answer composition (H1-18).

Model-free: the client seam is monkeypatched with canned CompletionResults, so
the prompt construction, citation parsing, accounting passthrough and fail-loud
guards are pinned without a network call. The generation itself is verified
LIVE against the local corpus (the backlog's done-when: a live query's
citations are all in the retrieved set).
"""

from __future__ import annotations

import logging

import pytest

import wealthlens_analyst.answer.compose as compose
from wealthlens_analyst.answer.compose import ComposedAnswer, compose_answer, strip_unresolved_citation_markers
from wealthlens_analyst.llm.client import CompletionResult
from wealthlens_analyst.retrieval.fts import ChunkHit


def _hit(chunk_id: int) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        document_id=f"doc-{chunk_id}",
        section=f"section-{chunk_id}",
        page=None,
        span=f"span-{chunk_id}",
        text=f"evidence text {chunk_id}",
        rank=1,
        score=1.0,
    )


_EVIDENCE = [_hit(9118), _hit(9140)]


class _FakeClient:
    def __init__(self, text: str, calls: dict[str, object]) -> None:
        self._text = text
        self._calls = calls

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
        self._calls.update({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        return CompletionResult(
            text=self._text,
            model="fake-model",
            tokens_in=1500,
            tokens_out=200,
            cost_gbp=1.6e-3,
        )


def _compose_with(monkeypatch: pytest.MonkeyPatch, text: str) -> tuple[ComposedAnswer, dict[str, object]]:
    calls: dict[str, object] = {}
    monkeypatch.setattr(compose, "get_client", lambda: _FakeClient(text, calls))
    answer = compose_answer("who holds the most wealth?", _EVIDENCE)
    return answer, calls


def test_prompt_carries_question_and_every_evidence_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _compose_with(monkeypatch, "answer [chunk:9118]")
    prompt = str(calls["prompt"])
    assert "who holds the most wealth?" in prompt
    # Each chunk appears as the exact marker the model must reuse, with its
    # provenance and verbatim text.
    assert "[chunk:9118] source=source-9118 document=doc-9118 section=section-9118" in prompt
    assert "[chunk:9140] source=source-9140 document=doc-9140 section=section-9140" in prompt
    assert "evidence text 9118" in prompt
    assert "evidence text 9140" in prompt
    # The hard constraints ride in the system message, not the user prompt.
    system = str(calls["system"])
    assert "[chunk:<id>]" in system
    assert calls["max_tokens"] == compose._MAX_ANSWER_TOKENS


def test_citations_parsed_in_first_appearance_order_deduplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    answer, _ = _compose_with(
        monkeypatch,
        "Top decile holds most [chunk:9140]. The distribution is skewed [chunk:9118], "
        "and the top band dominates [chunk:9140].",
    )
    assert answer.cited_chunk_ids == [9140, 9118]


def test_accounting_passes_through_from_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    answer, _ = _compose_with(monkeypatch, "cited [chunk:9118]")
    assert answer.tokens_in == 1500
    assert answer.tokens_out == 200
    assert answer.cost_gbp == pytest.approx(1.6e-3)
    assert answer.model == "fake-model"


def test_fabricated_citation_is_returned_but_logged_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Compose records what the answer CLAIMS (resolution is H1-19), but a
    # citation outside the evidence set is the product's core failure mode
    # and must be visible.
    with caplog.at_level(logging.WARNING, logger="wealthlens_analyst.answer.compose"):
        answer, _ = _compose_with(monkeypatch, "made up [chunk:777] but also real [chunk:9118]")
    assert answer.cited_chunk_ids == [777, 9118]
    assert any("777" in record.getMessage() for record in caplog.records)


def test_empty_evidence_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compose, "get_client", lambda: _FakeClient("x", {}))
    with pytest.raises(ValueError, match="abstention gate"):
        compose_answer("a question", [])


def test_blank_question_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compose, "get_client", lambda: _FakeClient("x", {}))
    with pytest.raises(ValueError, match="non-blank"):
        compose_answer("   ", _EVIDENCE)


def test_empty_generation_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    # A reasoning model can burn the whole output budget thinking; an empty
    # "answer" must be an error, not a silently served blank.
    with pytest.raises(RuntimeError, match="empty text"):
        _compose_with(monkeypatch, "   ")


def test_length_truncated_generation_fails_loud_with_recoverable_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    # finish_reason="length" means the answer was CUT OFF at the output cap —
    # a truncated figure must never be served as complete, and the raised
    # error must still carry the accounting for H1-20's error row.
    class _TruncatingClient:
        def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
            return CompletionResult(
                text="The top band holds 36.1% of gains and the threshold is £2,0",
                model="fake-model",
                tokens_in=1500,
                tokens_out=900,
                cost_gbp=3.2e-3,
                finish_reason="length",
            )

    monkeypatch.setattr(compose, "get_client", lambda: _TruncatingClient())
    with pytest.raises(compose.EmptyGenerationError, match="truncated") as excinfo:
        compose_answer("a question", _EVIDENCE)
    assert excinfo.value.result.cost_gbp == pytest.approx(3.2e-3)
    assert excinfo.value.result.tokens_out == 900


def test_empty_generation_error_carries_the_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compose, "get_client", lambda: _FakeClient("   ", {}))
    with pytest.raises(compose.EmptyGenerationError, match="empty text") as excinfo:
        compose_answer("a question", _EVIDENCE)
    # The exception is the record: the caller can still account the burn.
    assert excinfo.value.result.tokens_in == 1500
    assert excinfo.value.result.cost_gbp == pytest.approx(1.6e-3)


def test_fabricated_ids_are_machine_readable_on_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    answer, _ = _compose_with(monkeypatch, "made up [chunk:777] but also real [chunk:9118]")
    assert answer.fabricated_chunk_ids == [777]
    assert answer.cited_chunk_ids == [777, 9118]


def test_citation_format_drift_is_logged_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # "[chunk: 9140]" (space) fails the strict parser; that must be VISIBLE,
    # not a silently uncited claim.
    with caplog.at_level(logging.WARNING, logger="wealthlens_analyst.answer.compose"):
        answer, _ = _compose_with(monkeypatch, "strict [chunk:9118] but drifted [chunk: 9140] and [CHUNK=9118]")
    assert answer.cited_chunk_ids == [9118]
    assert any("format drift" in record.getMessage() for record in caplog.records)


def test_impossible_chunk_id_is_dropped_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An id beyond BIGINT can never exist and would crash a downstream DB
    # bind (H1-19) — dropped at parse, visibly.
    with caplog.at_level(logging.WARNING, logger="wealthlens_analyst.answer.compose"):
        answer, _ = _compose_with(monkeypatch, "real [chunk:9118] fake [chunk:99999999999999999999]")
    assert answer.cited_chunk_ids == [9118]
    assert any("BIGINT" in record.getMessage() for record in caplog.records)


def test_zero_citation_answer_is_flagged_not_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Legit for the mandated refusal sentence; the empty cited_chunk_ids list
    # is the machine-readable signal H1-20/21 act on — but it must be VISIBLE.
    with caplog.at_level(logging.WARNING, logger="wealthlens_analyst.answer.compose"):
        answer, _ = _compose_with(monkeypatch, "The evidence does not support an answer to this question.")
    assert answer.cited_chunk_ids == []
    assert any("ZERO citations" in record.getMessage() for record in caplog.records)


def test_strip_keeps_resolved_markers_and_removes_unresolved_ones() -> None:
    # 2 is a served citation (kept); 99 is pruned (marker + its leading space go).
    text = "Real [chunk:2] but also fake [chunk:99]."
    assert strip_unresolved_citation_markers(text, {2}) == "Real [chunk:2] but also fake."


def test_strip_removes_all_markers_when_none_resolve() -> None:
    assert strip_unresolved_citation_markers("A [chunk:1] and B [chunk:2].", set()) == "A and B."


def test_strip_is_a_no_op_when_every_marker_resolves() -> None:
    text = "A [chunk:1] and B [chunk:2]."
    assert strip_unresolved_citation_markers(text, {1, 2}) == text


def test_strip_handles_out_of_range_id_without_crashing() -> None:
    # A > BIGINT hallucination compose drops from cited_chunk_ids can still sit
    # inline; int() parses the arbitrary-width id and it is never resolved.
    text = "X [chunk:99999999999999999999999999]."
    assert strip_unresolved_citation_markers(text, {2}) == "X."


def test_strip_handles_adjacent_markers() -> None:
    # No whitespace between markers: only the resolved one survives.
    assert strip_unresolved_citation_markers("[chunk:1][chunk:2]", {2}) == "[chunk:2]"


def test_strip_leaves_text_without_markers_untouched() -> None:
    assert strip_unresolved_citation_markers("no markers here", {2}) == "no markers here"


def test_strip_removes_format_drift_near_miss_markers() -> None:
    # A near-miss ([chunk: 99] with a space, [CHUNK=99], [chunk = 7]) is never
    # counted as a citation by the STRICT parser, so its id is never in
    # resolved_ids -> the serving policy must strip it, not leak it as a dangling
    # unbacked reference (the security-lens finding this closes).
    assert (
        strip_unresolved_citation_markers("Real [chunk:2] and drift [chunk: 99].", {2}) == "Real [chunk:2] and drift."
    )
    assert strip_unresolved_citation_markers("A [CHUNK=99] b", set()) == "A b"
    assert strip_unresolved_citation_markers("A [chunk = 7] b", set()) == "A b"
