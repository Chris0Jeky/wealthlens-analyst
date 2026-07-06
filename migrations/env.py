"""Alembic environment for the WealthLens Analyst.

Migrations are HAND-WRITTEN (no autogenerate): the schema is small and each
revision documents its rationale, so deterministic, reviewable DDL beats
metadata diffing here. DATABASE_URL (env) overrides the alembic.ini fallback,
matching how the app itself is configured (config.py).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.environ.get("DATABASE_URL")
if database_url:
    # configparser interpolation: a literal % in the URL (e.g. percent-encoded
    # password) must be escaped or set_main_option raises ValueError.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

# Hand-written migrations: no model metadata to diff against.
target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a DB connection (alembic upgrade --sql)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
