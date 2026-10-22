"""Deterministic eval checks — the cheap checks that catch expensive failures.

Offline and CI-safe: validate the golden set and recorded /ask responses.
By default, exercise response checks on explicitly SYNTHETIC fixtures, never
golden answers. Use --responses to score a recording against REVIEWED golden
questions; DRAFT entries are counted and skipped for refusal correctness.

Recording format: recorded_responses.schema.json. Response bodies also validate
against api/schemas.py models and the published ask_response.schema.json.
Citation checks require canonical inline ids to match the response citation list
and source ids to belong to the app's corpus registry. They cannot verify database
existence or provenance truth offline.

Latency/cost checks are STUB bounds over recorded per-request measurements:
120,000 ms and GBP 1.00 by default (inclusive, deliberately generous). Override
with --max-latency-ms / --max-cost-gbp. Cost must include embedding + generation
as recorded by /ask accounting; these checks do not estimate or cap spend.
Live recording is PENDING: a future recorder must use guarded POST /ask and
its query_log accounting, never call a provider directly.

Usage:
    python evals/checks/deterministic.py            # golden + synthetic checks (CI)
    python evals/checks/deterministic.py --responses recorded.json
    python evals/checks/deterministic.py --live     # fails clearly: pending, no spend

Exit code 0 on success, 1 on any failure (fail loud, never skip silently).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import TypeAdapter, ValidationError

from wealthlens_analyst.answer.citations import load_source_registry
from wealthlens_analyst.answer.compose import _CITATION_NEAR_RE, _CITATION_RE, _MAX_CHUNK_ID
from wealthlens_analyst.api.schemas import AskResponse

CHECKS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = CHECKS_DIR.parent / "golden"
GOLDEN_SET = GOLDEN_DIR / "golden_set.jsonl"
GOLDEN_SCHEMA = GOLDEN_DIR / "golden_set.schema.json"
MIN_REFUSAL_PROBES = 5

#: The published /ask product-response schema (api/schemas.py is the source of
#: truth; this file is generated from it and drift-locked by a test).
ASK_RESPONSE_SCHEMA = CHECKS_DIR / "ask_response.schema.json"
RECORDING_SCHEMA = CHECKS_DIR / "recorded_responses.schema.json"
SYNTHETIC_RESPONSES = CHECKS_DIR / "fixtures" / "synthetic_responses.json"
SYNTHETIC_QUESTIONS = CHECKS_DIR / "fixtures" / "synthetic_questions.json"
SOURCE_REGISTRY = CHECKS_DIR.parents[1] / "data" / "corpus_manifest.yml"
DEFAULT_MAX_LATENCY_MS = 120_000.0
DEFAULT_MAX_COST_GBP = 1.0
_ASK_ADAPTER: TypeAdapter[Any] = TypeAdapter(AskResponse)


def load_golden_records() -> list[dict[str, Any]]:
    """Parse the golden JSONL; a malformed line is a hard failure."""
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(GOLDEN_SET.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"FAIL: {GOLDEN_SET.name} line {line_no} is not valid JSON: {exc}") from exc
    return records


def check_golden_static() -> list[str]:
    """Run the static golden-set checks; return a list of failure messages."""
    failures: list[str] = []
    schema = json.loads(GOLDEN_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    records = load_golden_records()

    for record in records:
        for error in validator.iter_errors(record):
            failures.append(f"{record.get('id', '<no id>')}: schema violation: {error.message}")

    ids = [r.get("id") for r in records]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        failures.append(f"duplicate golden ids: {sorted(str(d) for d in duplicates)}")

    refusal_probes = [r for r in records if r.get("category") == "out_of_corpus"]
    if len(refusal_probes) < MIN_REFUSAL_PROBES:
        failures.append(f"only {len(refusal_probes)} out-of-corpus refusal probes; need >= {MIN_REFUSAL_PROBES}")

    reviewed = [r for r in records if r.get("status") == "REVIEWED"]
    for record in reviewed:
        if record.get("expected_behaviour") == "answer":
            if not record.get("expected_answer"):
                failures.append(f"{record['id']}: REVIEWED answer question has empty expected_answer")
            if not record.get("required_citations"):
                failures.append(f"{record['id']}: REVIEWED answer question has no required_citations")
        else:  # refuse: a refusal probe must not smuggle in ground-truth text
            if record.get("expected_answer"):
                failures.append(f"{record['id']}: REVIEWED refuse question must have empty expected_answer")
            if record.get("required_citations"):
                failures.append(f"{record['id']}: REVIEWED refuse question must have no required_citations")

    # DRAFT records MUST keep expected_answer/required_citations empty: a human flips
    # status to REVIEWED only after authoring real ground truth, so a DRAFT carrying
    # a (machine-fabricated) answer or citation defeats the audit trail. Enforce the
    # "Empty while status is DRAFT" invariant the schema only describes in prose
    # (analyst NEVER-DO: never fabricate golden answers/citations of any kind).
    for record in records:
        if record.get("status") == "DRAFT":
            if record.get("expected_answer"):
                failures.append(f"{record.get('id', '<no id>')}: DRAFT question must have empty expected_answer")
            if record.get("required_citations"):
                failures.append(f"{record.get('id', '<no id>')}: DRAFT question must have no required_citations")

    drafts = len(records) - len(reviewed)
    print(
        f"golden set: {len(records)} records ({len(reviewed)} reviewed, {drafts} draft, {len(refusal_probes)} refusal probes)"
    )
    return failures


def _check_ask_response(
    body: dict[str, Any], validator: jsonschema.Draft202012Validator, source_ids: set[str] | None = None
) -> list[str]:
    """Validate one recorded /ask body against the contract and its invariants.

    Schema validity is the H1-20 done-when — and the schema already enforces
    ``mode`` (the discriminator) and, for an answer, ``citations`` minItems:1.
    Cross-field invariants mirror serving: every citation-shaped token must be
    canonical and within BIGINT, at least one must exist, and inline ids must
    equal listed citation ids. Every citation source must be in the app registry.
    """
    schema_failures = [f"schema violation: {error.message}" for error in validator.iter_errors(body)]
    if schema_failures:
        # A schema-invalid body: report that; don't assert semantics on top.
        return schema_failures
    try:
        # JSON-mode validation respects wire dates and catches constraints (such
        # as valid calendar dates) that JSON Schema without format checking misses.
        _ASK_ADAPTER.validate_json(json.dumps(body))
    except ValidationError:
        return ["schema violation: response rejected by api/schemas.py models"]
    failures: list[str] = []
    if body.get("mode") == "answer":
        if source_ids is None:
            source_ids = set(load_source_registry(SOURCE_REGISTRY))
        cited_ids = {citation["chunk_id"] for citation in body.get("citations", [])}
        markers: set[int] = set()
        for token in _CITATION_NEAR_RE.finditer(body["answer"]):
            match = _CITATION_RE.fullmatch(token.group())
            if match is None or len(match.group(1)) > 19 or int(match.group(1)) > _MAX_CHUNK_ID:
                failures.append(
                    f"answer body leaks unresolved citation markers (invalid citation marker): {token.group()[:128]!r}"
                )
                continue
            markers.add(int(match.group(1)))
        if not markers:
            failures.append("answer body has zero citation markers")
        orphan_markers = sorted(markers - cited_ids)
        if orphan_markers:
            failures.append(f"answer body leaks unresolved citation markers: {orphan_markers}")
        uncited_ids = sorted(cited_ids - markers)
        if uncited_ids:
            failures.append(f"answer body has uncited citation ids: {uncited_ids}")
        unknown_sources = sorted({citation["source_id"] for citation in body["citations"]} - source_ids)
        if unknown_sources:
            failures.append(f"answer citations have unknown corpus source ids: {unknown_sources}")
    return failures


def check_recorded_responses(
    recording: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
    max_cost_gbp: float = DEFAULT_MAX_COST_GBP,
) -> list[str]:
    """Check observations, then compare modes for the complete REVIEWED set.

    Expectations come from question metadata, never from the response recording.
    Synthetic question metadata is used only by the default CI smoke fixture.
    Missing REVIEWED responses, duplicate ids and unknown ids fail closed.
    """
    for name, bound in (("max_latency_ms", max_latency_ms), ("max_cost_gbp", max_cost_gbp)):
        if not math.isfinite(bound) or bound <= 0:
            return [f"{name}: bound must be finite and positive"]
    recording_schema = json.loads(RECORDING_SCHEMA.read_text(encoding="utf-8"))
    failures = [
        f"recording schema violation: {error.message}"
        for error in jsonschema.Draft202012Validator(recording_schema).iter_errors(recording)
    ]
    if failures:
        return failures
    schema = json.loads(ASK_RESPONSE_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    source_ids = set(load_source_registry(SOURCE_REGISTRY))
    by_id = {q["id"]: q for q in questions}
    seen: set[str] = set()
    checked: set[str] = set()
    for record in recording["records"]:
        question_id = record["question_id"]
        if question_id in seen:
            failures.append(f"{question_id}: duplicate response id")
        seen.add(question_id)
        body = record["response"]
        failures.extend(f"{question_id}: {failure}" for failure in _check_ask_response(body, validator, source_ids))
        for metric, bound in (("latency_ms", max_latency_ms), ("cost_gbp", max_cost_gbp)):
            value = record[metric]
            if value > bound or not math.isfinite(value):
                failures.append(f"{question_id}: STUB {metric} bound exceeded or non-finite (limit {bound})")
        question = by_id.get(question_id)
        if question is None:
            failures.append(f"{question_id}: unknown question id")
            continue
        if body.get("question") != question["question"]:
            failures.append(f"{question_id}: response question does not match question metadata")
        if question["status"] == "DRAFT":
            continue
        checked.add(question_id)
        expected = (
            "refusal"
            if question["category"] == "out_of_corpus" or question["expected_behaviour"] == "refuse"
            else "answer"
        )
        if body.get("mode") != expected:
            failures.append(f"{question_id}: expected {expected}, got {body.get('mode')}")

    reviewed = {q["id"] for q in questions if q["status"] == "REVIEWED"}
    failures.extend(f"{qid}: missing REVIEWED response" for qid in sorted(reviewed - seen))
    drafts = sum(q["status"] == "DRAFT" for q in questions)
    refusals = sum(
        q["status"] == "REVIEWED" and (q["category"] == "out_of_corpus" or q["expected_behaviour"] == "refuse")
        for q in questions
    )
    print(
        f"refusal correctness: {len(checked)}/{len(reviewed)} REVIEWED checked "
        f"({refusals} refusal, {len(reviewed) - refusals} answer expected), {drafts} DRAFT skipped"
    )
    print(f"recorded responses: {len(recording['records'])}; STUB bounds: {max_latency_ms} ms, GBP {max_cost_gbp}")
    return failures


def check_live() -> list[str]:
    """Pending guarded /ask recorder; deliberately performs no model calls."""
    return ["--live recording is PENDING; future implementation must use guarded /ask with query_log accounting"]


def _positive_bound(value: str) -> float:
    bound = float(value)
    if not math.isfinite(bound) or bound <= 0:
        raise argparse.ArgumentTypeError("bound must be finite and positive")
    return bound


def _reject_non_json_number(value: str) -> Any:
    raise ValueError(f"non-JSON number: {value}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_non_json_number)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="PENDING guarded /ask recording (fails without spending)")
    mode.add_argument("--responses", type=Path, help="offline recording to check against REVIEWED golden metadata")
    parser.add_argument("--max-latency-ms", type=_positive_bound, default=DEFAULT_MAX_LATENCY_MS)
    parser.add_argument("--max-cost-gbp", type=_positive_bound, default=DEFAULT_MAX_COST_GBP)
    args = parser.parse_args()

    failures = check_golden_static()
    if args.live:
        failures.extend(check_live())
    else:
        try:
            recording = _load_json(args.responses or SYNTHETIC_RESPONSES)
            questions = load_golden_records() if args.responses else _load_json(SYNTHETIC_QUESTIONS)["questions"]
            if isinstance(recording, dict) and recording.get("synthetic") is not (args.responses is None):
                failures.append("recording synthetic flag must be true for fixtures and false for golden scoring")
            failures.extend(
                check_recorded_responses(
                    recording, questions, max_latency_ms=args.max_latency_ms, max_cost_gbp=args.max_cost_gbp
                )
            )
            if not args.responses:
                golden = load_golden_records()
                drafts = sum(q["status"] == "DRAFT" for q in golden)
                print("SYNTHETIC fixture only: no golden response correctness claimed")
                print(
                    f"golden response checks: PENDING --responses ({len(golden) - drafts} REVIEWED, {drafts} DRAFT skipped)"
                )
        except (OSError, ValueError) as exc:
            failures.append(f"cannot load recording: {type(exc).__name__}")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("deterministic checks: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
