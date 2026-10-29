import jwt

from app.settings import get_settings

CREDS = {"email": "a@example.com", "password": "password123"}


def signup(client, **overrides):
    return client.post("/auth/signup", json={**CREDS, **overrides})


def test_signup_returns_auth_response(client):
    r = signup(client)
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"token", "userId", "kind", "plan"}
    assert (body["kind"], body["plan"]) == ("account", "free")
    claims = jwt.decode(body["token"], get_settings().jwt_secret, algorithms=["HS256"])
    assert claims["sub"] == body["userId"] and claims["plan"] == "free"


def test_duplicate_signup_is_409_case_insensitive(client):
    signup(client)
    r = signup(client, email="A@Example.com")
    assert r.status_code == 409
    assert r.json() == {"code": "conflict", "message": "Email already in use"}


def test_login_returns_same_user(client):
    user_id = signup(client).json()["userId"]
    r = client.post("/auth/login", json=CREDS)
    assert r.status_code == 200 and r.json()["userId"] == user_id


def test_unknown_email_and_wrong_password_look_identical(client):
    signup(client)
    unknown = client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"})
    wrong = client.post("/auth/login", json={**CREDS, "password": "wrong-password"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"code": "unauthorized", "message": "Invalid email or password"}


def test_invalid_input_is_400_bad_request(client):
    for body in ({"email": "not-an-email", "password": "password123"}, {**CREDS, "password": "short"}):
        r = client.post("/auth/signup", json=body)
        assert r.status_code == 400 and r.json()["code"] == "bad_request"


def test_bearer_token_authenticates(client):
    body = signup(client).json()
    r = client.get("/_test/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert r.status_code == 200 and r.json() == {"userId": body["userId"]}


def test_missing_or_bad_token_is_401(client):
    assert client.get("/_test/me").json()["code"] == "unauthorized"
    forged = jwt.encode({"sub": "x"}, "some-other-secret-that-is-long-enough!!", algorithm="HS256")
    r = client.get("/_test/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"


def test_unexpected_error_uses_contract_shape(client, monkeypatch):
    import app.auth

    def boom(*_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app.auth, "create_token", boom)
    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        r = c.post("/auth/signup", json=CREDS)
    assert r.status_code == 500
    assert r.json() == {"code": "internal", "message": "Internal server error"}


def test_health_and_unknown_route(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found"
