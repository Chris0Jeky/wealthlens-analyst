"""Fetch the frozen corpus documents listed in data/corpus_manifest.yml.

Downloads each entry's ``url`` into ``data/raw/<source_id>.<ext>``, computes
the file's sha256, and compares it to the manifest:

- manifest sha256 == literal ``PENDING``: the digest is printed so it can be
  recorded in the manifest (this script never edits the manifest itself);
- manifest sha256 present and DIFFERENT: fail loudly (exit 1) — a silently
  changed upstream document would poison the frozen corpus;
- manifest sha256 present and equal: verified, nothing to do.

Usage:
    python scripts/fetch_corpus.py            # fetch + verify everything
    python scripts/fetch_corpus.py --only ID  # a single source_id
    python scripts/fetch_corpus.py --verify   # verify existing files, no downloads
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "corpus_manifest.yml"
RAW_DIR = REPO_ROOT / "data" / "raw"
# Some publisher CDNs reject the default urllib User-Agent outright.
USER_AGENT = "wealthlens-analyst-corpus-fetch/1.0 (research; contact via repo)"
_EXT_BY_FORMAT = {"ods": ".ods", "xlsx": ".xlsx", "report": ".pdf"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_path(entry: dict[str, Any]) -> Path:
    ext = _EXT_BY_FORMAT.get(str(entry.get("format", "")), "")
    return RAW_DIR / f"{entry['source_id']}{ext}"


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="SOURCE_ID", help="fetch a single entry")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify already-downloaded files only; never download",
    )
    args = parser.parse_args()

    documents = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]
    if args.only:
        documents = [d for d in documents if d["source_id"] == args.only]
        if not documents:
            print(f"ERROR: no manifest entry with source_id={args.only}", file=sys.stderr)
            return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    pending: list[str] = []

    for entry in documents:
        source_id = entry["source_id"]
        target = _target_path(entry)
        if not target.exists():
            if args.verify:
                failures.append(f"{source_id}: missing {target.name} (run without --verify)")
                continue
            print(f"fetching {source_id} <- {entry['url']}")
            try:
                _download(str(entry["url"]), target)
            except Exception as exc:  # noqa: BLE001 - report every fetch failure loudly
                failures.append(f"{source_id}: download failed: {exc}")
                continue

        digest = _sha256(target)
        recorded = str(entry.get("sha256", "PENDING"))
        if recorded == "PENDING":
            pending.append(f"{source_id}: sha256: {digest}")
        elif recorded != digest:
            failures.append(
                f"{source_id}: sha256 MISMATCH — manifest {recorded[:12]}..., "
                f"file {digest[:12]}... (upstream changed or download corrupt)"
            )
        else:
            print(f"verified {source_id} ({digest[:12]}...)")

    if pending:
        print("\nPENDING digests — record these in data/corpus_manifest.yml:")
        for line in pending:
            print(f"  {line}")
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
