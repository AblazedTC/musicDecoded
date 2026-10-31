import jwt
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app.db import Base, engine
from app.settings import get_settings


def anon(client, device="device-1", platform="ios"):
    return client.post("/auth/anonymous", json={"deviceId": device, "platform": platform})


def test_anonymous_returns_guest_token(client):
    r = anon(client)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"token", "userId", "kind", "plan"}
    assert (body["kind"], body["plan"]) == ("guest", "free")
    claims = jwt.decode(body["token"], get_settings().jwt_secret, algorithms=["HS256"])
    assert (claims["sub"], claims["kind"], claims["plan"]) == (body["userId"], "guest", "free")


def test_anonymous_is_idempotent_per_device(client):
    first = anon(client).json()["userId"]
    assert anon(client, platform="android").json()["userId"] == first
    assert anon(client, device="device-2").json()["userId"] != first


def test_guest_token_authenticates(client):
    body = anon(client).json()
    r = client.get("/_test/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert r.status_code == 200 and r.json() == {"userId": body["userId"]}


def test_anonymous_validates_input(client):
    for body in ({"deviceId": "d"}, {"deviceId": "", "platform": "ios"},
                 {"deviceId": "d", "platform": "windows"}, {"deviceId": "x" * 129, "platform": "ios"}):
        r = client.post("/auth/anonymous", json=body)
        assert r.status_code == 400 and r.json()["code"] == "bad_request"


def test_guests_cannot_log_in_with_a_password(client):
    anon(client)
    r = client.post("/auth/login", json={"email": "a@example.com", "password": "dummy-password"})
    assert r.status_code == 401


def test_migrations_match_models(client):
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []
