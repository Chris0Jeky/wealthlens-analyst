"""Tests for the published /ask response schema (H1-20).

Pins three things without a database or a model call:
- the committed evals/checks/ask_response.schema.json is regenerated from the
  Pydantic models (drift-lock — the two must never diverge silently);
- the discriminated union resolves each body to the right variant by ``mode``;
- the deterministic checker's per-response invariants (schema validity + the
  answer serving-policy: >= 1 citation, no leaked marker) bite as intended.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import TypeAdapter, ValidationError

from wealthlens_analyst.api.schemas import (
    AnswerResponse,
    AskResponse,
    OverBudgetResponse,
    RefusalResponse,
    ask_response_json_schema,
)

_SCHEMA_FILE = Path(__file__).resolve().parents[1] / "evals" / "checks" / "ask_response.schema.json"

# The eval checker lives outside the package (evals/checks), imported by path as
# the sibling golden-guard test does.
_CHECKS = Path(__file__).resolve().parents[1] / "evals" / "checks"
sys.path.insert(0, str(_CHECKS))

import deterministic as det  # noqa: E402  (path-dependent import of the eval checker)

_ADAPTER: TypeAdapter[Any] = TypeAdapter(AskResponse)


def _sample_answer() -> dict[str, Any]:
    return {
        "mode": "answer",
        "question": "q",
        "answer": "Top decile holds most [chunk:9140].",
        "citations": [
            {
                "chunk_id": 9140,
                "source_id": "hmrc-cgt-statistics",
                "source_name": "HMRC CGT statistics",
                "document_id": "doc",
                "section": "sec",
                "page": None,
                "span": "span",
                "url": "https://example.org/cgt",
                "access_date": "2026-06-01",
            }
        ],
    }


def test_committed_schema_matches_the_generated_schema() -> None:
    """Drift-lock: the committed JSON schema equals what the models generate.

    If a response model changes, this fails until the committed file is
    regenerated (python -c 'from wealthlens_analyst.api.schemas import
    ask_response_json_schema; ...'), so the published contract can never
    silently diverge from the code.
    """
    committed = json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))
    assert committed == ask_response_json_schema()


def test_union_discriminates_each_variant_by_mode() -> None:
    assert isinstance(_ADAPTER.validate_python(_sample_answer()), AnswerResponse)
    assert isinstance(_ADAPTER.validate_python({"mode": "refusal", "question": "q", "reason": "no"}), RefusalResponse)
    assert isinstance(
        _ADAPTER.validate_python({"mode": "over_budget", "question": "q", "reason": "cap"}), OverBudgetResponse
    )


def test_union_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"mode": "mystery", "question": "q", "reason": "x"})


def test_committed_schema_validates_variants_and_rejects_malformed_bodies() -> None:
    validator = jsonschema.Draft202012Validator(json.loads(_SCHEMA_FILE.read_text(encoding="utf-8")))
    assert list(validator.iter_errors(_sample_answer())) == []
    assert list(validator.iter_errors({"mode": "refusal", "question": "q", "reason": "no"})) == []
    assert list(validator.iter_errors({"mode": "over_budget", "question": "q", "reason": "cap"})) == []
    # An answer missing its required fields, and an unknown mode, must not validate.
    assert list(validator.iter_errors({"mode": "answer", "question": "q"}))
    assert list(validator.iter_errors({"mode": "mystery", "question": "q", "reason": "x"}))
    # The discriminator is required: a mode-less answer-shaped body is invalid.
    assert list(validator.iter_errors({k: v for k, v in _sample_answer().items() if k != "mode"}))
    # An answer means a CITED answer: empty citations is rejected (minItems: 1).
    assert list(validator.iter_errors({**_sample_answer(), "citations": []}))


def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(_SCHEMA_FILE.read_text(encoding="utf-8")))


def test_check_ask_response_passes_a_well_formed_answer() -> None:
    assert det._check_ask_response(_sample_answer(), _validator()) == []


def test_check_ask_response_passes_a_refusal() -> None:
    assert det._check_ask_response({"mode": "refusal", "question": "q", "reason": "no"}, _validator()) == []


def test_check_ask_response_rejects_an_answer_with_no_citations() -> None:
    # citations is minItems:1 in the schema, so an empty-citations answer is
    # rejected at the schema layer (an answer means a fully-cited answer).
    body = {**_sample_answer(), "citations": []}
    assert det._check_ask_response(body, _validator())


def test_check_ask_response_flags_a_leaked_orphan_marker() -> None:
    # A [chunk:99] marker in the body with 99 NOT among the served citations is
    # the serving-policy violation the check exists to catch.
    body = {**_sample_answer(), "answer": "Real [chunk:9140] and leaked [chunk:99]."}
    failures = det._check_ask_response(body, _validator())
    assert any("leaks unresolved citation markers" in f and "99" in f for f in failures)


def test_check_ask_response_flags_a_leaked_near_miss_marker() -> None:
    # The live check's marker regex is lenient, so a format-drift leak
    # ("[chunk: 99]") is caught too — not just the strict form.
    body = {**_sample_answer(), "answer": "Real [chunk:9140] and drift [chunk: 99]."}
    failures = det._check_ask_response(body, _validator())
    assert any("leaks unresolved citation markers" in f and "99" in f for f in failures)


def test_check_ask_response_reports_schema_violation_for_a_malformed_body() -> None:
    failures = det._check_ask_response({"mode": "answer", "question": "q"}, _validator())
    assert failures and all("schema violation" in f for f in failures)
