"""Deterministic eval checks — the cheap checks that catch expensive failures.

Two layers:

1. STATIC (implemented now, runs in CI on every PR): the golden set parses,
   every record validates against golden_set.schema.json, ids are unique,
   and the refusal probe set is present (>= 5 out-of-corpus questions).
2. LIVE (pending H1-23, behind --live): runs against a serving /ask —
   citation presence + resolvability, response schema validity, correct
   refusal on every out-of-corpus golden question, latency/cost bounds.

Usage:
    python evals/checks/deterministic.py            # static checks (CI)
    python evals/checks/deterministic.py --live     # + live checks (H1-23)

Exit code 0 on success, 1 on any failure (fail loud, never skip silently).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema

CHECKS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = CHECKS_DIR.parent / "golden"
GOLDEN_SET = GOLDEN_DIR / "golden_set.jsonl"
GOLDEN_SCHEMA = GOLDEN_DIR / "golden_set.schema.json"
MIN_REFUSAL_PROBES = 5

#: The published /ask product-response schema (api/schemas.py is the source of
#: truth; this file is generated from it and drift-locked by a test).
ASK_RESPONSE_SCHEMA = CHECKS_DIR / "ask_response.schema.json"

#: One answerable (in-corpus) and one unanswerable (out-of-corpus) probe for the
#: live schema check. The corpus is the frozen WAS + HMRC slice: a capital-gains
#: concentration question is answerable; a geography question is not. `expected`
#: is advisory (logged) — model-dependent refusal correctness is H1-23's golden
#: set; H1-20 asserts only that live responses validate against the schema.
_LIVE_PROBES: tuple[tuple[str, str], ...] = (
    ("Which size-of-gain band accounts for the largest share of capital gains?", "answer"),
    ("What is the capital city of France?", "refusal"),
)

#: Citation-shaped marker, LENIENT (mirrors answer.compose._CITATION_STRIP_RE):
#: catches strict "[chunk:9140]" AND format-drift near-misses ("[chunk: 99]",
#: "[CHUNK=99]") so the live check flags a leaked drift marker, not just a strict
#: one. Any matched id not among the served citations is an orphan (policy leak).
_MARKER_RE = re.compile(r"\[\s*chunk\s*[:=]?\s*(\d+)\s*\]", re.IGNORECASE)


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


def _check_ask_response(body: dict[str, Any], validator: jsonschema.Draft202012Validator) -> list[str]:
    """Validate one live /ask response body against the schema and its invariants.

    Schema validity is the H1-20 done-when — and the schema already enforces
    ``mode`` (the discriminator) and, for an answer, ``citations`` minItems:1.
    The one invariant the schema cannot express is cross-field: an ``answer``
    body must not carry an inline ``[chunk:<id>]`` marker (strict OR drift form)
    for an id that is not among its served citations (the serving policy —
    fabricated / pruned / drift markers are stripped before serving).
    """
    schema_failures = [f"schema violation: {error.message}" for error in validator.iter_errors(body)]
    if schema_failures:
        # A schema-invalid body: report that; don't assert semantics on top.
        return schema_failures
    failures: list[str] = []
    if body.get("mode") == "answer":
        cited_ids = {citation["chunk_id"] for citation in body.get("citations", [])}
        markers = {int(match) for match in _MARKER_RE.findall(body.get("answer", ""))}
        orphan_markers = sorted(markers - cited_ids)
        if orphan_markers:
            failures.append(f"answer body leaks unresolved citation markers: {orphan_markers}")
    return failures


def check_live() -> list[str]:
    """Validate live /ask responses against the published schema (H1-20 done-when).

    Builds the app in-process (real engine + registry + model), sends an
    answerable and an unanswerable probe, and checks each response validates
    against ask_response.schema.json (plus the answer serving-policy invariants).
    Needs a reachable DATABASE_URL + real provider config in the environment, and
    it SPENDS (one embedding + up to one gpt-5.4-mini generation per probe).

    H1-23 EXTENDS this with the full golden refusal set (correct refusal on every
    out-of-corpus golden question) and latency/cost bounds; H1-20 covers schema
    validity only.
    """
    from fastapi.testclient import TestClient

    from wealthlens_analyst.api.app import create_app

    schema = json.loads(ASK_RESPONSE_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    failures: list[str] = []
    with TestClient(create_app()) as client:
        for question, expected in _LIVE_PROBES:
            response = client.post("/ask", json={"question": question})
            if response.status_code != 200:
                failures.append(f"{question!r}: expected 200, got {response.status_code}: {response.text[:200]}")
                continue
            body = response.json()
            print(f"live /ask {question!r}: mode={body.get('mode')} (expected {expected})")
            failures.extend(f"{question!r}: {failure}" for failure in _check_ask_response(body, validator))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also run live checks against /ask")
    args = parser.parse_args()

    failures = check_golden_static()
    if args.live:
        failures.extend(check_live())

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("deterministic checks: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
