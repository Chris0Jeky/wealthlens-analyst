"""The abstention gate: refuse honestly when the corpus can't answer.

When fused retrieval evidence is weak, /ask returns a structured
"cannot answer from this corpus" refusal instead of a generated guess.
Refusal is a visible product feature with its own response schema and its own
deterministic eval check (the 5+ out-of-corpus golden questions MUST refuse).

Gating mechanism: ADR 0003 D4 = threshold on the fused RRF score plus a
min-hits guard (the cheapest mechanism deterministic checks can test). The
mechanism stays behind one function (``evaluate_evidence``) so any future
change is a swap, not a refactor.

H1-21 implements the gate with fixture-tested INITIAL thresholds; wiring it into
/ask ahead of generation, and calibrating the thresholds against the reviewed
golden set, is H1-22 (which needs the golden set, H1-02, Chris-blocked).
"""

from __future__ import annotations

from dataclasses import dataclass

from wealthlens_analyst.retrieval.fts import ChunkHit
from wealthlens_analyst.retrieval.fuse_rrf import RRF_K

# Stable machine-readable reason served in the refusal body.
REFUSAL_REASON_WEAK_EVIDENCE = "cannot_answer_from_corpus"

# --- Gate thresholds (ADR 0003 D4: fused-RRF-score threshold + min-hits) ------
# Refuse when the top fused RRF score < _MIN_TOP_SCORE, OR fewer than _MIN_HITS
# chunks score >= _MIN_HIT_SCORE (weak or thin evidence). Zero model calls,
# fully deterministic.
#
# These are INITIAL, UNCALIBRATED values. ADR 0003 D4 calibrates them ONCE
# against the reviewed golden set's in-corpus vs out-of-corpus separation — that
# is H1-22's job, and the golden set is still Chris-blocked (H1-02). Until then
# they are deliberately LENIENT (favour answering): H1-20 already refuses when
# the model cites nothing that resolves, so that citation check — not this gate —
# is the primary out-of-corpus backstop today, and a too-tight threshold here
# would suppress good answers. The values are anchored to RRF(k=RRF_K) over two
# retrievers, where a single-list rank-1 chunk scores 1/(k+1) and a chunk in
# BOTH lists scores more; the gate mainly catches the degenerate near-empty-fusion
# case before it spends a generation. Retuning is a one-line change behind this
# single function (the mechanism is intentionally swappable).
_MIN_TOP_SCORE = 1.0 / (RRF_K + 1)  # top chunk must clear a single-list rank-1 floor
_MIN_HITS = 2  # need corroboration, not one lucky hit
_MIN_HIT_SCORE = 1.0 / (RRF_K + 25)  # a "supporting" hit bar (~rank-25 in one list)


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the confidence gate for one question."""

    answerable: bool
    # The signal the gate used (the top fused RRF score today; a rerank score or
    # judge verdict if D4 is ever revisited), logged to query_log.gate_signal so
    # refusal behaviour is auditable.
    signal: float
    reason: str | None  # set when answerable is False


def evaluate_evidence(question: str, evidence: list[ChunkHit]) -> GateDecision:
    """Decide whether the retrieved evidence supports answering at all.

    ADR 0003 D4: refuse when the fused evidence is weak — the top fused RRF score
    is below _MIN_TOP_SCORE, OR fewer than _MIN_HITS chunks clear _MIN_HIT_SCORE.
    Zero model calls, fully deterministic (testable in CI with fixture chunks).

    ``evidence`` is the RRF-FUSED hit list — each ``hit.score`` is its fused RRF
    score (pass the fused top-N). ``signal`` is the top fused score (0.0 for no
    evidence), independent of list order (taken as the max), and is what H1-22
    logs to ``query_log.gate_signal``. The gate runs BEFORE generation, so a
    refusal here costs zero generation spend (ADR 0003 D4's whole point).
    """
    if not question.strip():
        raise ValueError("evaluate_evidence requires a non-blank question")
    top_score = max((hit.score for hit in evidence), default=0.0)
    strong_hits = sum(1 for hit in evidence if hit.score >= _MIN_HIT_SCORE)
    answerable = top_score >= _MIN_TOP_SCORE and strong_hits >= _MIN_HITS
    reason = None if answerable else REFUSAL_REASON_WEAK_EVIDENCE
    return GateDecision(answerable=answerable, signal=top_score, reason=reason)
