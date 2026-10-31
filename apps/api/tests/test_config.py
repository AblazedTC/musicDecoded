from app.db import SessionLocal
from app.kv import rate_key
from app.models import User
from app.settings import Settings, get_settings

from conftest import bearer


def test_config_requires_auth(client):
    assert client.get("/config").status_code == 401


def test_guest_config_defaults(client, guest):
    headers, body = guest
    r = client.get("/config", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"offlineStreamAudio": False, "soundcloudEnabled": False, "maxUploadBytes": 104857600,
                        "plan": "free", "limits": {"dailyJobs": 10, "jobsUsedToday": 0}}


def test_jobs_used_today_comes_from_redis(client, guest, redis_fake):
    headers, body = guest
    redis_fake.set(rate_key(body["userId"]), 3)
    assert client.get("/config", headers=headers).json()["limits"]["jobsUsedToday"] == 3


def test_account_and_pro_limits(client):
    token = client.post("/auth/signup", json={"email": "a@example.com", "password": "password123"}).json()["token"]
    assert client.get("/config", headers=bearer(token)).json()["limits"]["dailyJobs"] == 50
    with SessionLocal() as db:
        db.query(User).update({User.plan: "pro"})
        db.commit()
    body = client.get("/config", headers=bearer(token)).json()
    assert body["plan"] == "pro" and body["limits"]["dailyJobs"] is None


def test_flags_come_from_env(client, guest, monkeypatch):
    monkeypatch.setattr(get_settings(), "offline_stream_audio", True)
    monkeypatch.setattr(get_settings(), "soundcloud_enabled", True)
    body = client.get("/config", headers=guest[0]).json()
    assert body["offlineStreamAudio"] is True and body["soundcloudEnabled"] is True


def test_limit_env_parsing(monkeypatch):
    monkeypatch.setenv("LIMIT_PRO", "")
    monkeypatch.setenv("LIMIT_GUEST", "3")
    s = Settings()
    assert (s.limit_guest, s.limit_free, s.limit_pro) == (3, 50, None)
