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

Usage:
    python evals/checks/check_citations_live.py ["your question"]

Exit code 0 when (a) the real answer has >= 1 citation and all of them resolve
to full provenance with a registry source name + URL, and (b) an injected
fabricated chunk_id is stripped and reported unresolved; 1 otherwise (fail
loud, never skip silently).
"""

from __future__ import annotations

import dataclasses
import logging
import sys

from wealthlens_analyst.answer.citations import resolve_citations
from wealthlens_analyst.answer.compose import compose_answer
from wealthlens_analyst.retrieval.dense import search_dense
from wealthlens_analyst.retrieval.fts import search_fts
from wealthlens_analyst.retrieval.fuse_rrf import fuse_rrf

_DEFAULT_QUESTION = (
    "Which size-of-gain band accounts for the largest share of capital gains, "
    "and how concentrated are gains among taxpayers?"
)

#: An id far beyond any real chunk, to prove a fabricated citation is stripped.
_FABRICATED_ID = 99_999_999


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    question = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_QUESTION

    lexical = search_fts(question)
    dense = search_dense(question)
    fused = fuse_rrf(lexical, dense)
    print(f"retrieved: fts={len(lexical)} dense={len(dense)} fused={len(fused)}")
    if not fused:
        print("FAIL: retrieval returned no candidates (is the corpus ingested?)")
        return 1

    answer = compose_answer(question, fused)
    print("\n=== ANSWER ===")
    print(answer.text)
    print(f"\ncited={answer.cited_chunk_ids} fabricated={answer.fabricated_chunk_ids}")
    if not answer.cited_chunk_ids:
        print("FAIL: the answer carries no citations to resolve")
        return 1

    # (a) The real answer's citations resolve to full provenance.
    resolved = resolve_citations(answer)
    print("\n=== RESOLVED CITATIONS ===")
    for citation in resolved.citations:
        loc = f"page {citation.page}" if citation.page is not None else (citation.section or "-")
        print(
            f"  [chunk:{citation.chunk_id}] {citation.source_name} ({citation.source_id}) — {loc}"
            f"\n      {citation.url} (accessed {citation.access_date})"
        )
    if resolved.unresolved_chunk_ids:
        print(f"unresolved (unexpected for the clean answer): {resolved.unresolved_chunk_ids}")

    non_fabricated = [cid for cid in answer.cited_chunk_ids if cid not in answer.fabricated_chunk_ids]
    resolved_ids = [c.chunk_id for c in resolved.citations]
    if not resolved.citations:
        print("FAIL: no citation resolved")
        return 1
    if resolved_ids != non_fabricated:
        print(f"FAIL: resolved ids {resolved_ids} != evidence-grounded cited ids {non_fabricated}")
        return 1
    if any(not c.source_name or not c.url.startswith("http") for c in resolved.citations):
        print("FAIL: a resolved citation is missing a registry source name or URL")
        return 1

    # (b) A fabricated citation is caught and stripped.
    tampered = dataclasses.replace(
        answer,
        cited_chunk_ids=[*answer.cited_chunk_ids, _FABRICATED_ID],
        fabricated_chunk_ids=[*answer.fabricated_chunk_ids, _FABRICATED_ID],
    )
    tampered_resolved = resolve_citations(tampered)
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
