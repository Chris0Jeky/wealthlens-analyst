"""Runtime configuration, read from environment variables.

Everything tunable lives here so behaviour is visible in one place.
Defaults are conservative: the reranker is OFF (repo rule: new features
default OFF) and the budget is fail-closed (no cap configured -> no spend).
See `.env.example` for the full variable list.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var ("true"/"1"/"yes", case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the service configuration."""

    database_url: str
    anthropic_api_key: str
    analyst_model: str
    embedding_model: str
    openai_api_key: str  # embeddings provider (ADR 0003 D2)
    rerank_enabled: bool
    cohere_api_key: str
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    # Hard monthly spend cap in GBP (ADR 0002). None means NOT configured,
    # which the budget middleware treats as "block all spend" (fail-closed).
    budget_monthly_cap_gbp: float | None
    app_env: str


def _parse_budget_cap(raw: str | None) -> float | None:
    """Parse the spend cap; a malformed value fails LOUDLY at startup.

    The cap is the one variable whose whole job is safety. Silently treating a
    typo as "no cap" would block all spend with no explanation (confusing) —
    and silently treating it as "unlimited" would be dangerous. So: clear
    startup error, no guessing.

    A cap must be a POSITIVE, FINITE number. ``float()`` alone accepts "nan",
    "inf" and negatives, which would defeat the cap — "nan" is the worst, because
    every ``spend < nan`` comparison is False, so the middleware would never trip
    and spend would be unbounded. Reject all of those loudly here (the sole
    validation seam), matching the fail-closed posture (ADR 0002).
    """
    if not raw:
        return None  # not configured -> fail-closed (middleware blocks spend)
    try:
        cap = float(raw)
    except ValueError as exc:
        raise ValueError(f"BUDGET_MONTHLY_CAP_GBP must be numeric, got {raw!r}") from exc
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"BUDGET_MONTHLY_CAP_GBP must be a positive, finite number, got {raw!r}")
    return cap


def load_settings() -> Settings:
    """Build Settings from the process environment."""
    cap_raw = os.environ.get("BUDGET_MONTHLY_CAP_GBP")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        analyst_model=os.environ.get("ANALYST_MODEL", ""),
        embedding_model=os.environ.get("EMBEDDING_MODEL", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        rerank_enabled=_env_bool("RERANK_ENABLED", default=False),
        cohere_api_key=os.environ.get("COHERE_API_KEY", ""),
        langfuse_host=os.environ.get("LANGFUSE_HOST", ""),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        budget_monthly_cap_gbp=_parse_budget_cap(cap_raw),
        app_env=os.environ.get("APP_ENV", "development"),
    )
