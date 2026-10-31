"""T-601: RevenueCat webhook → users.plan, effective_plan expiry, require_pro."""
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import jwt
import pytest
from fastapi import Depends

from app.billing import require_pro
from app.db import SessionLocal
from app.main import app
from app.models import BillingEvent, User, effective_plan
from app.settings import get_settings

from conftest import bearer

FIXTURES = Path(__file__).parent / "fixtures" / "revenuecat"
SECRET = "rc-webhook-secret-for-tests"
HOUR_MS = 3600 * 1000


@app.get("/_test/pro")
def _pro(user: Annotated[User, Depends(require_pro)]) -> dict:
    return {"userId": str(user.id)}


@pytest.fixture(autouse=True)
def webhook_secret(monkeypatch):
    monkeypatch.setattr(get_settings(), "revenuecat_webhook_secret", SECRET)


@pytest.fixture
def account(client):
    body = client.post("/auth/signup", json={"email": "pro@example.com", "password": "password123"}).json()
    return bearer(body["token"]), body["userId"]


def now_ms() -> int:
    return int(time.time() * 1000)


def payload(name: str, user_id: str, **event) -> dict:
    """A recorded-shape fixture, retargeted at user_id, with a fresh event id unless one is given."""
    body = json.loads((FIXTURES / f"{name}.json").read_text())
    body["event"].update(app_user_id=user_id, id=str(uuid.uuid4()), **event)
    return body


def send(client, body: dict, secret: str | None = SECRET):
    headers = {} if secret is None else {"Authorization": f"Bearer {secret}"}
    return client.post("/billing/revenuecat", json=body, headers=headers)


def user_row(user_id: str) -> User:
    with SessionLocal() as db:
        return db.get(User, uuid.UUID(user_id))


def plan_of(client, headers) -> str:
    return client.get("/config", headers=headers).json()["plan"]


@pytest.mark.parametrize("name", ["initial_purchase", "renewal", "product_change", "uncancellation"])
def test_grant_events_make_the_user_pro_until_expiry(client, account, name):
    headers, user_id = account
    expires = now_ms() + 24 * HOUR_MS
    assert send(client, payload(name, user_id, expiration_at_ms=expires)).status_code == 200
    user = user_row(user_id)
    assert user.plan == "pro"
    assert abs(user.plan_expires_at.replace(tzinfo=timezone.utc).timestamp() * 1000 - expires) < 1000
    config = client.get("/config", headers=headers).json()
    assert config["plan"] == "pro" and config["limits"]["dailyJobs"] is None


def test_non_expiring_purchase(client, account):
    headers, user_id = account
    send(client, payload("initial_purchase", user_id, expiration_at_ms=None))
    assert user_row(user_id).plan_expires_at is None and plan_of(client, headers) == "pro"


def test_cancellation_keeps_pro_until_expiry(client, account):
    headers, user_id = account
    expires = now_ms() + HOUR_MS
    send(client, payload("initial_purchase", user_id, expiration_at_ms=expires))
    assert send(client, payload("cancellation", user_id, expiration_at_ms=expires)).status_code == 200
    assert plan_of(client, headers) == "pro"


def test_expiration_downgrades_to_free(client, account):
    headers, user_id = account
    expires = now_ms() + HOUR_MS
    send(client, payload("initial_purchase", user_id, expiration_at_ms=expires))
    assert send(client, payload("expiration", user_id, expiration_at_ms=expires)).status_code == 200
    assert user_row(user_id).plan == "free"
    config = client.get("/config", headers=headers).json()
    assert config["plan"] == "free" and config["limits"]["dailyJobs"] == 50


def test_stale_expiration_after_a_renewal_is_ignored(client, account):
    headers, user_id = account
    send(client, payload("renewal", user_id, expiration_at_ms=now_ms() + 30 * 24 * HOUR_MS))
    send(client, payload("expiration", user_id, expiration_at_ms=now_ms() - HOUR_MS))  # delivered late
    assert plan_of(client, headers) == "pro"


def test_out_of_order_renewal_never_shortens_expiry(client, account):
    _, user_id = account
    later = now_ms() + 60 * 24 * HOUR_MS
    send(client, payload("renewal", user_id, expiration_at_ms=later))
    send(client, payload("renewal", user_id, expiration_at_ms=now_ms() + HOUR_MS))
    expires = user_row(user_id).plan_expires_at.replace(tzinfo=timezone.utc)
    assert abs(expires.timestamp() * 1000 - later) < 1000


def test_billing_issue_keeps_pro_through_grace_period(client, account):
    headers, user_id = account
    expires, grace = now_ms() + HOUR_MS, now_ms() + 16 * 24 * HOUR_MS
    send(client, payload("initial_purchase", user_id, expiration_at_ms=expires))
    body = payload("billing_issue", user_id, expiration_at_ms=expires, grace_period_expiration_at_ms=grace)
    assert send(client, body).status_code == 200
    user = user_row(user_id)
    assert user.plan == "pro"
    assert abs(user.plan_expires_at.replace(tzinfo=timezone.utc).timestamp() * 1000 - grace) < 1000


def test_redelivered_event_is_applied_once(client, account):
    headers, user_id = account
    purchase = payload("initial_purchase", user_id, expiration_at_ms=now_ms() + HOUR_MS)
    expiration = payload("expiration", user_id, expiration_at_ms=now_ms() + HOUR_MS)
    send(client, purchase)
    send(client, expiration)
    assert send(client, purchase).status_code == 200  # RevenueCat retry of an old event
    assert plan_of(client, headers) == "free"
    with SessionLocal() as db:
        assert db.query(BillingEvent).count() == 2


def test_unhandled_event_type_is_acknowledged_and_ignored(client, account):
    headers, user_id = account
    assert send(client, payload("test", user_id)).status_code == 200
    assert user_row(user_id).plan == "free"


@pytest.mark.parametrize("app_user_id", [str(uuid.uuid4()), "$RCAnonymousID:12345678901234567890123456789012"])
def test_unknown_user_is_acknowledged_and_ignored(client, account, app_user_id):
    r = send(client, payload("initial_purchase", app_user_id, expiration_at_ms=now_ms() + HOUR_MS))
    assert r.status_code == 200
    assert user_row(account[1]).plan == "free"
    with SessionLocal() as db:
        assert db.query(BillingEvent).count() == 1


@pytest.mark.parametrize("secret", [None, "wrong", ""])
def test_bad_secret_is_401(client, account, secret):
    body = payload("initial_purchase", account[1], expiration_at_ms=now_ms() + HOUR_MS)
    r = send(client, body, secret)
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"
    assert user_row(account[1]).plan == "free"


def test_user_jwt_is_not_the_webhook_secret(client, account):
    body = payload("initial_purchase", account[1], expiration_at_ms=now_ms() + HOUR_MS)
    r = client.post("/billing/revenuecat", json=body, headers=account[0])
    assert r.status_code == 401


def test_bad_secret_is_checked_before_the_body(client):
    assert send(client, {"nope": 1}, "wrong").status_code == 401
    r = send(client, {"nope": 1})
    assert r.status_code == 400 and r.json()["code"] == "bad_request"


def test_unconfigured_webhook_is_404(client, account, monkeypatch):
    monkeypatch.setattr(get_settings(), "revenuecat_webhook_secret", "")
    r = send(client, payload("initial_purchase", account[1]), "")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_expired_pro_is_free_in_config_tokens_and_daily_limit(client, account, monkeypatch):
    headers, user_id = account
    monkeypatch.setattr(get_settings(), "limit_free", 1)
    monkeypatch.setattr(get_settings(), "limit_pro", 3)
    # Pro whose expiry passed with no EXPIRATION event delivered (yet).
    send(client, payload("initial_purchase", user_id, expiration_at_ms=now_ms() - 1000))
    assert user_row(user_id).plan == "pro"
    config = client.get("/config", headers=headers).json()
    assert config["plan"] == "free" and config["limits"]["dailyJobs"] == 1

    login = client.post("/auth/login", json={"email": "pro@example.com", "password": "password123"}).json()
    assert login["plan"] == "free"
    assert jwt.decode(login["token"], options={"verify_signature": False})["plan"] == "free"

    assert client.post("/jobs", headers=headers, json={"source": "youtube", "videoId": "video000001"}).status_code == 202
    r = client.post("/jobs", headers=headers, json={"source": "youtube", "videoId": "video000002"})
    assert r.status_code == 429 and r.json()["code"] == "rate_limited"


def test_new_tokens_carry_the_pro_claim(client, account):
    _, user_id = account
    send(client, payload("initial_purchase", user_id, expiration_at_ms=now_ms() + HOUR_MS))
    login = client.post("/auth/login", json={"email": "pro@example.com", "password": "password123"}).json()
    assert login["plan"] == "pro"
    assert jwt.decode(login["token"], options={"verify_signature": False})["plan"] == "pro"


def test_require_pro(client, account):
    headers, user_id = account
    r = client.get("/_test/pro", headers=headers)
    assert r.status_code == 403 and r.json()["code"] == "pro_required"
    send(client, payload("initial_purchase", user_id, expiration_at_ms=now_ms() + HOUR_MS))
    assert client.get("/_test/pro", headers=headers).status_code == 200
    assert client.get("/_test/pro").status_code == 401


def test_effective_plan():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert effective_plan(User(plan="free"), now) == "free"
    assert effective_plan(User(plan="pro", plan_expires_at=None), now) == "pro"
    assert effective_plan(User(plan="pro", plan_expires_at=now + timedelta(seconds=1)), now) == "pro"
    assert effective_plan(User(plan="pro", plan_expires_at=now), now) == "free"
    naive = (now + timedelta(hours=1)).replace(tzinfo=None)  # as SQLite returns it
    assert effective_plan(User(plan="pro", plan_expires_at=naive), now) == "pro"
