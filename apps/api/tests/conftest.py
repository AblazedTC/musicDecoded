import os
import tempfile
from pathlib import Path

# Settings are read at import time, so configure the environment before importing the app.
_db = Path(tempfile.mkdtemp()) / "test.db"
os.environ["APP_ENV"] = "prod"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"
os.environ["DATABASE_URL"] = f"sqlite:///{_db.as_posix()}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from typing import Annotated  # noqa: E402
from fastapi import Depends  # noqa: E402

from app.auth import current_user  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402


@app.get("/_test/me")
def _me(user: Annotated[User, Depends(current_user)]) -> dict:
    return {"userId": str(user.id)}


@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:  # runs lifespan → create_all
        yield c
