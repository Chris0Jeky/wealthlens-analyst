"""Corpus ingestion: the frozen slice -> chunks with ingestion-time provenance.

The slice (ONS WAS, HMRC distributional statistics, 3-5 IFS/RF reports) is
defined by the documents in data/corpus_manifest.yml and is
FROZEN until v1 ships. A chunk with null provenance fails ingestion.
"""
