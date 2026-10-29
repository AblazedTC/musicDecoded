import os
import tempfile
from pathlib import Path

# Settings are read at import time, so configure the environment before importing the app.
_db = Path(tempfile.mkdtemp()) / "test.db"
os.environ["APP_ENV"] = "prod"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"
os.environ["DATABASE_URL"] = f"sqlite:///{_db.as_posix()}"

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from typing import Annotated  # noqa: E402
from fastapi import Depends  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.auth import current_user  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.kv import get_redis  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402


@app.get("/_test/me")
def _me(user: Annotated[User, Depends(current_user)]) -> dict:
    return {"userId": str(user.id)}


def _reset_db() -> None:
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))


@pytest.fixture
def redis_fake():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def client(redis_fake):
    _reset_db()
    app.dependency_overrides[get_redis] = lambda: redis_fake
    with TestClient(app) as c:  # runs lifespan → alembic upgrade head
        yield c
    app.dependency_overrides.clear()


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def guest(client):
    """Auth headers + body for a fresh guest."""
    body = client.post("/auth/anonymous", json={"deviceId": "device-1", "platform": "ios"}).json()
    return bearer(body["token"]), body
