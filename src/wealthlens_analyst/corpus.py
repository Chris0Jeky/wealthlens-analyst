"""Read the frozen corpus identity from the repository-native manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def default_manifest_path() -> Path:
    """Find the manifest, honoring the deployed WEALTHLENS_REPO_ROOT override."""
    override = os.environ.get("WEALTHLENS_REPO_ROOT")
    if override:
        return Path(override) / "data" / "corpus_manifest.yml"
    for parent in Path(__file__).resolve().parents:
        path = parent / "data" / "corpus_manifest.yml"
        if path.is_file():
            return path
    raise RuntimeError(
        "could not locate data/corpus_manifest.yml in any ancestor directory; set WEALTHLENS_REPO_ROOT to override"
    )


def load_corpus_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Index every manifest document; malformed or duplicate identities fail loudly."""
    manifest_path = path or default_manifest_path()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} is not a YAML mapping")
    entries = data.get("documents")
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_path} has no top-level 'documents' list")
    documents: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip() for key in ("source_id", "title", "url")
        ):
            raise ValueError(f"{manifest_path}: document entry missing source_id/title/url: {entry!r}")
        source_id = entry["source_id"]
        if source_id in documents:
            raise ValueError(f"{manifest_path}: duplicate source_id {source_id!r}")
        documents[source_id] = entry
    return documents
