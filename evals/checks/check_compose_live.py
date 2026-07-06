"""Re-runnable live check for the H1-18 done-when (NOT run in CI).

The backlog's acceptance for H1-18 is a LIVE property: "a live local query
returns an answer whose citations are all in the retrieved set". This script
is the committed, re-runnable form of that check, so the acceptance evidence
does not live only in a PR description. It needs the local analyst-db
(:15432, ingested corpus) and a real OPENAI_API_KEY in the environment, and
it SPENDS: one query embedding + one gpt-5.4-mini generation (well under a
penny at the verified prices in llm/client.py).

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

from wealthlens_analyst.answer.compose import compose_answer
from wealthlens_analyst.retrieval.dense import search_dense
from wealthlens_analyst.retrieval.fts import search_fts
from wealthlens_analyst.retrieval.fuse_rrf import fuse_rrf

_DEFAULT_QUESTION = (
    "Which size-of-gain band accounts for the largest share of capital gains, "
    "and how concentrated are gains among taxpayers?"
)


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
    print("\n=== ACCOUNTING ===")
    print(
        f"model={answer.model} tokens_in={answer.tokens_in} "
        f"tokens_out={answer.tokens_out} cost_gbp={answer.cost_gbp:.8f}"
    )
    print(f"cited={answer.cited_chunk_ids} fabricated={answer.fabricated_chunk_ids}")

    if not answer.cited_chunk_ids:
        print("FAIL: the answer carries no citations")
        return 1
    if answer.fabricated_chunk_ids:
        print(f"FAIL: citations outside the retrieved set: {answer.fabricated_chunk_ids}")
        return 1
    print("OK: every citation is in the retrieved set (H1-18 done-when holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
