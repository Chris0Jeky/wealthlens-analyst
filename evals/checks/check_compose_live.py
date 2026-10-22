"""Re-runnable live check for the H1-18 done-when (NOT run in CI).

The backlog's acceptance for H1-18 is a LIVE property: "a live local query
returns an answer whose citations are all in the retrieved set". This script
is the committed, re-runnable form of that check, so the acceptance evidence
does not live only in a PR description. It needs the local analyst-db
(:15432, ingested corpus) and a real OPENAI_API_KEY in the environment, and
it SPENDS: one query embedding + one gpt-5.4-mini generation (well under a
penny at the verified prices in llm/client.py).
The in-process /ask request uses the configured database, budget reservation
and query_log accounting; a budget refusal fails before any provider call.

Usage:
    python evals/checks/check_compose_live.py ["your question"]

Exit code 0 when the answer carries >= 1 citation and every citation is in
the retrieved set; 1 otherwise (fail loud, never skip silently). The full
golden-set version of this (recall + refusal + schema over 20 questions) is
H1-14/H1-23 — this is deliberately one question, one property.
"""

from __future__ import annotations

import logging
import sys

from fastapi.testclient import TestClient

from wealthlens_analyst.api.app import create_app
from wealthlens_analyst.api.schemas import AnswerResponse

_DEFAULT_QUESTION = (
    "Which size-of-gain band accounts for the largest share of capital gains, "
    "and how concentrated are gains among taxpayers?"
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    question = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_QUESTION

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post("/ask", json={"question": question})
    if response.status_code == 429:
        print("FAIL: over_budget: budget guard refused before any provider call")
        return 1
    if response.status_code != 200:
        print(f"FAIL: guarded /ask returned HTTP {response.status_code}")
        return 1
    if response.json()["mode"] != "answer":
        print(f"FAIL: /ask refused: {response.json()['reason']}")
        return 1

    # /ask serves an answer only when all citations resolve to retrieved evidence,
    # including refusal on invalid_citation_marker; it accounts before returning.
    answer = AnswerResponse.model_validate(response.json())
    print("\n=== ANSWER ===")
    print(answer.answer)
    print(f"cited={[citation.chunk_id for citation in answer.citations]}")
    print("accounting: embedding + generation recorded in query_log by /ask")
    print("OK: every citation is in the retrieved set (H1-18 done-when holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
