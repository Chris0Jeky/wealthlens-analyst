"""Offline eval regression cases use SYNTHETIC observations, never golden answers."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest

_CHECKS = Path(__file__).resolve().parents[1] / "evals" / "checks"
sys.path.insert(0, str(_CHECKS))

import deterministic as det  # noqa: E402


def _inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recording = json.loads(det.SYNTHETIC_RESPONSES.read_text(encoding="utf-8"))
    questions = json.loads(det.SYNTHETIC_QUESTIONS.read_text(encoding="utf-8"))["questions"]
    return recording, questions


def test_recording_schema_and_synthetic_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    jsonschema.Draft202012Validator.check_schema(json.loads(det.RECORDING_SCHEMA.read_text(encoding="utf-8")))
    recording, questions = _inputs()
    assert det.check_recorded_responses(recording, questions) == []
    output = capsys.readouterr().out
    assert "2/2 REVIEWED checked (1 refusal, 1 answer expected), 1 DRAFT skipped" in output
    assert "STUB bounds: 120000.0 ms, GBP 1.0" in output


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"citations": []}, "schema violation"),
        ({"answer": "Synthetic orphan [chunk:999]."}, "unresolved citation markers: [999]"),
        ({"answer": "Synthetic drift orphan [CHUNK=999]."}, "invalid citation marker"),
        ({"mode": "unknown"}, "schema violation"),
    ],
)
def test_invalid_answer_fails(change: dict[str, Any], message: str) -> None:
    recording, questions = _inputs()
    recording["records"][0]["response"].update(change)
    assert any(message in failure for failure in det.check_recorded_responses(recording, questions))


@pytest.mark.parametrize(
    "marker",
    [
        "[chunk: 101]",
        "[CHUNK:101]",
        "[chunk:0101]",
        "[chunk:abc]",
        "[chunk:1",
        "[chunk:0]",
        f"[chunk:{2**63}]",
        pytest.param(f"[chunk:{'9' * 5000}]", id="oversized-id"),
    ],
)
def test_noncanonical_marker_fails_even_with_a_valid_citation(marker: str) -> None:
    recording, questions = _inputs()
    recording["records"][0]["response"]["answer"] = f"Synthetic valid [chunk:101] and invalid {marker}"
    assert any("invalid citation marker" in failure for failure in det.check_recorded_responses(recording, questions))


def test_answer_without_inline_markers_fails() -> None:
    recording, questions = _inputs()
    recording["records"][0]["response"]["answer"] = "Synthetic uncited answer."
    assert any("zero citation markers" in failure for failure in det.check_recorded_responses(recording, questions))


def test_uncited_citation_fails() -> None:
    recording, questions = _inputs()
    citations = recording["records"][0]["response"]["citations"]
    citations.append({**citations[0], "chunk_id": 102})
    assert any(
        "uncited citation ids: [102]" in failure for failure in det.check_recorded_responses(recording, questions)
    )


@pytest.mark.parametrize("citation_index", [0, 1])
def test_unknown_corpus_source_fails(citation_index: int) -> None:
    recording, questions = _inputs()
    body = recording["records"][0]["response"]
    body["answer"] = "Synthetic citations [chunk:101] [chunk:102]."
    body["citations"].append({**body["citations"][0], "chunk_id": 102})
    body["citations"][citation_index]["source_id"] = "synthetic-unknown-source"
    assert any(
        "unknown corpus source ids: ['synthetic-unknown-source']" in failure
        for failure in det.check_recorded_responses(recording, questions)
    )


def test_valid_canonical_markers_match_citation_set() -> None:
    recording, questions = _inputs()
    body = recording["records"][0]["response"]
    body["answer"] = "Synthetic repeated [chunk:101] [chunk:102] [chunk:101]."
    body["citations"].insert(0, {**body["citations"][0], "chunk_id": 102})
    assert det.check_recorded_responses(recording, questions) == []


def test_pydantic_models_reject_invalid_wire_date() -> None:
    recording, questions = _inputs()
    recording["records"][0]["response"]["citations"][0]["access_date"] = "2026-02-30"
    assert any("api/schemas.py models" in failure for failure in det.check_recorded_responses(recording, questions))


@pytest.mark.parametrize("mode", ["answer", "over_budget"])
def test_refusal_probe_must_be_an_actual_refusal(mode: str) -> None:
    recording, questions = _inputs()
    response = copy.deepcopy(recording["records"][0]["response"])
    response.update(mode=mode, question=questions[1]["question"], reason="Synthetic budget outcome")
    recording["records"][1]["response"] = response
    assert f"SYN-REFUSE: expected refusal, got {mode}" in det.check_recorded_responses(recording, questions)


@pytest.mark.parametrize("mode", ["refusal", "over_budget"])
def test_reviewed_in_corpus_cannot_refuse(mode: str) -> None:
    recording, questions = _inputs()
    recording["records"][0]["response"] = {
        "mode": mode,
        "question": questions[0]["question"],
        "reason": "Synthetic refusal",
    }
    assert f"SYN-ANSWER: expected answer, got {mode}" in det.check_recorded_responses(recording, questions)


def test_draft_modes_are_not_scored_even_for_refusal_probes(capsys: pytest.CaptureFixture[str]) -> None:
    recording, questions = _inputs()
    questions[0].update(status="DRAFT", category="out_of_corpus", expected_behaviour="refuse")
    assert det.check_recorded_responses(recording, questions) == []
    assert "2 DRAFT skipped" in capsys.readouterr().out
    # Skipping correctness does not bypass structural response validation.
    recording["records"][0]["response"]["citations"] = []
    assert any("schema violation" in failure for failure in det.check_recorded_responses(recording, questions))


def test_missing_reviewed_response_fails_but_missing_draft_does_not() -> None:
    recording, questions = _inputs()
    recording["records"].pop()
    assert det.check_recorded_responses(recording, questions) == []
    recording["records"].pop()
    assert "SYN-REFUSE: missing REVIEWED response" in det.check_recorded_responses(recording, questions)


def test_duplicate_unknown_and_mismatched_questions_fail() -> None:
    recording, questions = _inputs()
    recording["records"].append(copy.deepcopy(recording["records"][0]))
    recording["records"][1]["question_id"] = "SYN-UNKNOWN"
    recording["records"][0]["response"]["question"] = "Different synthetic question"
    failures = det.check_recorded_responses(recording, questions)
    assert "SYN-ANSWER: duplicate response id" in failures
    assert "SYN-UNKNOWN: unknown question id" in failures
    assert "SYN-ANSWER: response question does not match question metadata" in failures
    assert "SYN-REFUSE: missing REVIEWED response" in failures


@pytest.mark.parametrize("metric", ["latency_ms", "cost_gbp"])
def test_configured_bounds_are_inclusive_and_fail_on_excess(metric: str) -> None:
    recording, questions = _inputs()
    bounds = {"max_latency_ms": 10.0, "max_cost_gbp": 0.001}
    assert det.check_recorded_responses(recording, questions, **bounds) == []
    recording["records"][0][metric] *= 2
    assert any(
        f"STUB {metric} bound exceeded" in f for f in det.check_recorded_responses(recording, questions, **bounds)
    )


def test_large_numeric_measurement_flags_violation_without_float_overflow() -> None:
    recording, questions = _inputs()
    recording["records"][0]["latency_ms"] = 10**400
    assert any("STUB latency_ms bound exceeded" in f for f in det.check_recorded_responses(recording, questions))


@pytest.mark.parametrize("value", [-1, "1", None, float("nan"), float("inf")])
@pytest.mark.parametrize("metric", ["latency_ms", "cost_gbp"])
def test_invalid_or_non_finite_measurements_fail(metric: str, value: Any) -> None:
    recording, questions = _inputs()
    recording["records"][0][metric] = value
    assert det.check_recorded_responses(recording, questions)


@pytest.mark.parametrize("bound", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["max_latency_ms", "max_cost_gbp"])
def test_invalid_configuration_fails(name: str, bound: float) -> None:
    recording, questions = _inputs()
    assert det.check_recorded_responses(recording, questions, **{name: bound}) == [
        f"{name}: bound must be finite and positive"
    ]


@pytest.mark.parametrize("field", ["format_version", "synthetic", "records"])
def test_missing_envelope_fields_fail(field: str) -> None:
    recording, questions = _inputs()
    del recording[field]
    assert any("recording schema violation" in f for f in det.check_recorded_responses(recording, questions))


def test_missing_measurement_and_empty_recording_fail() -> None:
    recording, questions = _inputs()
    del recording["records"][0]["cost_gbp"]
    assert any("recording schema violation" in f for f in det.check_recorded_responses(recording, questions))
    recording["records"] = []
    assert det.check_recorded_responses(recording, questions)


def test_default_cli_is_offline_and_labels_synthetic_coverage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins

    real_import = builtins.__import__

    def offline_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"wealthlens_analyst.api.app", "openai", "anthropic", "cohere"}:
            pytest.fail(f"offline eval attempted to import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", offline_import)
    monkeypatch.setattr(sys, "argv", ["deterministic.py"])
    assert det.main() == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC fixture only: no golden response correctness claimed" in output
    assert "golden response checks: PENDING --responses" in output


def test_live_is_pending_and_fails_without_spending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["deterministic.py", "--live"])
    assert det.main() == 1
    assert "PENDING" in det.check_live()[0]


def test_cli_reads_configured_bound(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["deterministic.py", "--max-cost-gbp", "0.0005"])
    assert det.main() == 1
    assert "STUB cost_gbp bound exceeded" in capsys.readouterr().err


def test_responses_cli_scores_external_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recording, questions = _inputs()
    path = tmp_path / "recording.json"
    recording["synthetic"] = False  # Simulate a real recording using only synthetic test inputs.
    path.write_text(json.dumps(recording), encoding="utf-8")
    monkeypatch.setattr(det, "check_golden_static", lambda: [])
    monkeypatch.setattr(det, "load_golden_records", lambda: questions)
    monkeypatch.setattr(sys, "argv", ["deterministic.py", "--responses", str(path)])
    assert det.main() == 0
    recording["records"].pop(1)
    path.write_text(json.dumps(recording), encoding="utf-8")
    assert det.main() == 1
    assert "missing REVIEWED response" in capsys.readouterr().err


def test_synthetic_recording_cannot_be_scored_as_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["deterministic.py", "--responses", str(det.SYNTHETIC_RESPONSES)])
    assert det.main() == 1


@pytest.mark.parametrize("contents", ["{bad json", "NaN", "[]", "null"])
def test_bad_recording_file_fails_cleanly(contents: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "recording.json"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["deterministic.py", "--responses", str(path)])
    assert det.main() == 1
