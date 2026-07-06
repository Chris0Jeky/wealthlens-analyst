"""Lock the DRAFT-empty invariant in the golden-set deterministic checker.

The analyst NEVER-DO list forbids fabricating golden answers/citations; the
schema documents "expected_answer/required_citations empty while status is DRAFT"
but does not bind it, and check_golden_static() previously enforced it only for
REVIEWED records. So a DRAFT record carrying a machine-written answer passed
silently. These tests lock the new DRAFT guard (bites on a fabricated record;
stays green on the shipped all-empty DRAFT set).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CHECKS = Path(__file__).resolve().parents[1] / "evals" / "checks"
sys.path.insert(0, str(_CHECKS))

import deterministic as det  # noqa: E402  (path-dependent import of the eval checker)


def test_draft_record_with_fabricated_answer_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A DRAFT record carrying expected_answer/required_citations fails the check."""
    real = [json.loads(line) for line in det.GOLDEN_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    draft = next(r for r in real if r.get("status") == "DRAFT" and r.get("expected_behaviour") == "answer")
    fabricated = {**draft, "expected_answer": "FABRICATED machine answer", "required_citations": ["ons-was-wealth"]}

    fixture = tmp_path / "golden.jsonl"
    fixture.write_text(json.dumps(fabricated) + "\n", encoding="utf-8")
    monkeypatch.setattr(det, "GOLDEN_SET", fixture)

    failures = det.check_golden_static()
    assert any("DRAFT" in f and "expected_answer" in f for f in failures)
    assert any("DRAFT" in f and "required_citations" in f for f in failures)


def test_shipped_golden_set_passes_the_draft_guard() -> None:
    """Every shipped DRAFT record keeps its answer/citation fields empty."""
    draft_failures = [f for f in det.check_golden_static() if "DRAFT" in f]
    assert draft_failures == []
