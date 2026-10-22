"""Exercise the live eval entry points without a database or provider calls."""

from __future__ import annotations

import importlib
import sys
from datetime import date
from pathlib import Path

import pytest

import wealthlens_analyst.answer.compose as compose
from wealthlens_analyst.answer.citations import (
    ChunkProvenance,
    ResolvedCitations,
    SourceMeta,
    _resolve_citations,
)
from wealthlens_analyst.answer.compose import ComposedAnswer
from wealthlens_analyst.llm.client import CompletionResult
from wealthlens_analyst.retrieval.fts import ChunkHit


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> CompletionResult:
        return CompletionResult(self.text, "fake-model", 100, 20, 0.001, finish_reason="stop")


@pytest.mark.parametrize("check_name", ["check_compose_live", "check_citations_live"])
@pytest.mark.parametrize(
    ("text", "expected_exit"),
    [
        ("A [chunk:9118].", 0),
        ("A [chunk:9118]. Unsupported claim [chunk:99999999999999999999].", 1),
    ],
    ids=["valid", "mixed-markers"],
)
def test_live_check_rejects_invalid_markers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check_name: str,
    text: str,
    expected_exit: int,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "evals" / "checks"))
    check = importlib.import_module(check_name)
    monkeypatch.setattr(sys, "argv", [check_name])
    evidence = [ChunkHit(9118, "test-source", "test-document", "section", None, None, "evidence", 1, 1.0)]
    monkeypatch.setattr(check, "search_fts", lambda question: evidence)
    monkeypatch.setattr(check, "search_dense", lambda question: evidence)
    monkeypatch.setattr(compose, "get_client", lambda: _FakeClient(text))

    # Use the real resolver logic with canned DB provenance and registry data.
    def resolve(answer: ComposedAnswer) -> ResolvedCitations:
        return _resolve_citations(
            answer,
            {9118: ChunkProvenance(9118, "test-source", "test-document", "section", None, None, date(2026, 1, 1))},
            {"test-source": SourceMeta("Test source", "https://example.test/source")},
        )

    if check_name == "check_citations_live":
        monkeypatch.setattr(check, "resolve_citations", resolve)

    assert check.main() == expected_exit
    output = capsys.readouterr().out
    if expected_exit:
        assert "FAIL: invalid citation marker" in output
        assert "[chunk:99999999999999999999]" in output
        assert "OK:" not in output
    else:
        assert "OK:" in output
