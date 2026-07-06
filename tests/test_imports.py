"""Scaffolding smoke tests: every stub module imports cleanly.

This is the M0 guarantee — the package skeleton is importable (so the CI job,
mypy, and future tasks build on a sound base) while pending logic raises
NotImplementedError rather than silently doing nothing.
"""

from __future__ import annotations

import pytest


def test_package_imports() -> None:
    """All modules import without side effects or missing dependencies."""
    import wealthlens_analyst
    import wealthlens_analyst.answer.abstain
    import wealthlens_analyst.answer.citations
    import wealthlens_analyst.answer.compose
    import wealthlens_analyst.api.app
    import wealthlens_analyst.api.routes
    import wealthlens_analyst.budget.middleware
    import wealthlens_analyst.budget.models
    import wealthlens_analyst.config
    import wealthlens_analyst.ingest.fetch_documents
    import wealthlens_analyst.ingest.slice_corpus
    import wealthlens_analyst.llm.client
    import wealthlens_analyst.retrieval.dense
    import wealthlens_analyst.retrieval.fts
    import wealthlens_analyst.retrieval.fuse_rrf
    import wealthlens_analyst.retrieval.rerank

    assert wealthlens_analyst.__version__


def test_app_factory_builds() -> None:
    """The FastAPI app builds with its routes registered.

    /healthz now checks database reachability (H1-13), so its behaviour is
    covered in test_api.py with the engine dependency overridden — this smoke
    test only guarantees the factory itself needs no environment.
    """
    from wealthlens_analyst.api.app import create_app

    app = create_app()
    # The OpenAPI schema is the stable cross-version way to see registered
    # routes (app.routes introspection differs across fastapi/starlette).
    assert {"/ask", "/healthz", "/metrics/data"} <= set(app.openapi()["paths"])


def test_pending_stubs_raise_not_implemented() -> None:
    """Pending seams fail loudly, not silently (repo rule: no silent failures).

    Retrieval (H1-10/11/12), ingestion (H1-09), the debug route + accounting
    (H1-13/15), generation (H1-18), citation resolution (H1-19), the plain-mode
    schema (H1-20) and the abstention gate (H1-21) are implemented and pinned by
    their own test files. EVERY remaining pending seam is pinned here, so silently
    gutting a stub (e.g. an accidental `return []`) cannot survive the suite: the
    reranker (H1-16, deferred until a COHERE_API_KEY exists) and budget accounting
    (H1-27).
    """
    from wealthlens_analyst.budget.models import current_spend_gbp
    from wealthlens_analyst.retrieval.rerank import rerank

    with pytest.raises(NotImplementedError, match="H1-16"):
        rerank("q", [], top_k=1)
    with pytest.raises(NotImplementedError, match="H1-27"):
        current_spend_gbp()


def test_budget_config_fail_closed_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No BUDGET_MONTHLY_CAP_GBP in the environment -> cap is None (blocked)."""
    from wealthlens_analyst.config import load_settings

    monkeypatch.delenv("BUDGET_MONTHLY_CAP_GBP", raising=False)
    assert load_settings().budget_monthly_cap_gbp is None

    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "10.00")
    assert load_settings().budget_monthly_cap_gbp == 10.0

    # A malformed cap fails LOUDLY at startup — never silently unguarded.
    monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", "ten pounds")
    with pytest.raises(ValueError, match="BUDGET_MONTHLY_CAP_GBP"):
        load_settings()

    # float() accepts these (incl. mixed-case spellings — the realistic .env typo),
    # but they would DEFEAT the cap (nan never trips the spend check; inf/negative/
    # zero are nonsensical), so each must fail loudly.
    for bad in ("nan", "NaN", "inf", "Infinity", "-inf", "1e400", "-5", "0", "0.0"):
        monkeypatch.setenv("BUDGET_MONTHLY_CAP_GBP", bad)
        with pytest.raises(ValueError, match="BUDGET_MONTHLY_CAP_GBP"):
            load_settings()
