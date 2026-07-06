"""Fetch the report documents (IFS / Resolution Foundation PDFs) in the slice.

This is the repo's FIRST document pipeline — the existing 11 pipelines in
automation/data-pipelines/ fetch tabular data only. It follows their
conventions: source URL + access date + licence come from
registries/sources.yml (the entries added by task H1-01), downloads are
reproducible, and failures are loud (a moved/changed document must fail the
run, not silently ingest the wrong text).

Downloads land in data/corpus/ (gitignored — fetched, not committed).

Pending: task H1-06 in tasks/hero1-backlog.md.
"""

from __future__ import annotations

from pathlib import Path


def fetch_slice_documents(dest: Path) -> list[Path]:
    """Download every report-format slice source; return the local paths.

    Verifies each download against a recorded checksum where one exists and
    records the access date alongside the file.
    """
    raise NotImplementedError("H1-06: document fetching not yet implemented")
