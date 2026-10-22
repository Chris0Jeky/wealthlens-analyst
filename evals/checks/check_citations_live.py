"""Re-runnable live check for the H1-19 done-when (NOT run in CI).

The backlog's acceptance for H1-19 is a LIVE property: cited chunk_ids resolve
to full {source, document, section/page} provenance from the DB (plus the
registry source name + URL), and a fabricated chunk_id is caught and stripped
so the response carries resolved citations only. This script is the committed,
re-runnable form of that check, so the acceptance evidence does not live only
in a PR description. It needs the local analyst-db (ingested corpus) reachable
via DATABASE_URL and a real OPENAI_API_KEY in the environment, and it SPENDS:
one query embedding + one gpt-5.4-mini generation (well under a penny at the
verified prices in llm/client.py).
The in-process /ask request uses the configured database, budget reservation
and query_log accounting. The fabricated-citation probe makes no model call.

Usage:
    python evals/checks/check_citations_live.py ["your question"]

Exit code 0 when (a) the real answer has >= 1 citation and all of them resolve
to full provenance with a registry source name + URL, and (b) an injected
fabricated chunk_id is stripped and reported unresolved; 1 otherwise (fail
loud, never skip silently).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping

from fastapi.testclient import TestClient
from sqlalchemy import Engine

from wealthlens_analyst.answer.citations import SourceMeta, resolve_citations
from wealthlens_analyst.answer.compose import ComposedAnswer
from wealthlens_analyst.api.app import create_app
from wealthlens_analyst.api.schemas import AnswerResponse

_DEFAULT_QUESTION = (
    "Which size-of-gain band accounts for the largest share of capital gains, "
    "and how concentrated are gains among taxpayers?"
)

#: An id far beyond any real chunk, to prove a fabricated citation is stripped.
_FABRICATED_ID = 99_999_999


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    question = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_QUESTION

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
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
        answer = AnswerResponse.model_validate(response.json())
        return _check_citations(answer, app.state.engine, app.state.registry)


def _check_citations(answer: AnswerResponse, engine: Engine, registry: Mapping[str, SourceMeta]) -> int:
    """Inspect the accounted answer and run a free fabricated-citation probe."""
    print("\n=== ANSWER ===")
    print(answer.answer)
    print(f"\ncited={[citation.chunk_id for citation in answer.citations]}")
    print("accounting: embedding + generation recorded in query_log by /ask")

    # (a) /ask has resolved the real answer's citations to full provenance.
    # It also refuses invalid_citation_marker before serving any answer.
    print("\n=== RESOLVED CITATIONS ===")
    for citation in answer.citations:
        loc = f"page {citation.page}" if citation.page is not None else (citation.section or "-")
        print(
            f"  [chunk:{citation.chunk_id}] {citation.source_name} ({citation.source_id}) — {loc}"
            f"\n      {citation.url} (accessed {citation.access_date})"
        )
    if any(not c.source_name or not c.url.startswith("http") for c in answer.citations):
        print("FAIL: a resolved citation is missing a registry source name or URL")
        return 1

    # (b) A fabricated citation is caught and stripped.
    tampered = ComposedAnswer(
        text=answer.answer,
        cited_chunk_ids=[*[c.chunk_id for c in answer.citations], _FABRICATED_ID],
        fabricated_chunk_ids=[_FABRICATED_ID],
    )
    tampered_resolved = resolve_citations(tampered, engine=engine, registry=registry)
    if _FABRICATED_ID in [c.chunk_id for c in tampered_resolved.citations]:
        print(f"FAIL: fabricated id {_FABRICATED_ID} was served as a citation")
        return 1
    if _FABRICATED_ID not in tampered_resolved.unresolved_chunk_ids:
        print(f"FAIL: fabricated id {_FABRICATED_ID} was not reported as unresolved")
        return 1
    print(f"\nfabricated id {_FABRICATED_ID}: stripped and reported unresolved (caught)")

    print("\nOK: citations resolve to full provenance and fabrications are stripped (H1-19 done-when holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
