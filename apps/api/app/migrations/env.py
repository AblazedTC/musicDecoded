"""Alembic environment. Runs at API startup (app.db.migrate) and from the CLI (`uv run alembic ...`)."""
from alembic import context

import app.models  # noqa: F401  (registers every table on Base.metadata)
from app.db import Base, engine

target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      render_as_batch=connection.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


conn = context.config.attributes.get("connection")
if conn is not None:
    _run(conn)
else:
    with engine.connect() as conn:
        _run(conn)
