"""RAGAS evaluation runner (pending task H1-25).

Runs RAGAS metrics over the REVIEWED golden questions against a serving
/ask endpoint, with the judge LLM and embeddings injected through the
client seam configuration (no direct provider wiring here — ADR 0002).

Only REVIEWED records participate: scoring against DRAFT ground truth would
launder unreviewed text into an eval claim.

Output: a metrics dict consumed by `make eval-report`, which combines it
with the deterministic results and latency/cost percentiles into a
committed report under evals/reports/.

Usage (once implemented): python evals/run_ragas.py --out evals/reports/
"""

from __future__ import annotations

import sys


def run_ragas() -> dict[str, float]:
    """Run RAGAS over the reviewed golden subset; return metric scores."""
    raise NotImplementedError("H1-25: RAGAS wiring not yet implemented")


def main() -> int:
    raise NotImplementedError("H1-25: RAGAS runner CLI not yet implemented")


if __name__ == "__main__":
    sys.exit(main())
