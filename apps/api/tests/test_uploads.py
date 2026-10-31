import hashlib
import os
import uuid

import httpx
import pytest

from app.models import Upload
from app.db import SessionLocal

from conftest import bearer

SHA = "a" * 64


def new_upload(client, headers, **overrides):
    return client.post("/uploads", headers=headers,
                       json={"contentType": "audio/mpeg", "sizeBytes": 1234, "sha256": SHA, **overrides})


def test_create_upload_returns_presigned_put(client, guest):
    headers, me = guest
    r = new_upload(client, headers)
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"uploadId", "putUrl", "expiresAt"}
    assert body["putUrl"].startswith(f"https://s3.test/put/uploads/{body['uploadId']}?ct=audio/mpeg")
    with SessionLocal() as db:
        row = db.get(Upload, uuid.UUID(body["uploadId"]))
    assert (str(row.user_id), row.sha256, row.content_type, row.size_bytes, row.object_key) == \
        (me["userId"], SHA, "audio/mpeg", 1234, f"uploads/{body['uploadId']}")


def test_too_large_is_413(client, guest):
    assert new_upload(client, guest[0], sizeBytes=104857600).status_code == 201
    r = new_upload(client, guest[0], sizeBytes=104857601)
    assert r.status_code == 413 and r.json()["code"] == "too_large"


@pytest.mark.parametrize("bad", [{"contentType": "video/mp4"}, {"sizeBytes": 0}, {"sha256": "A" * 64},
                                 {"sha256": "abc"}, {"contentType": None}])
def test_invalid_upload_is_400(client, guest, bad):
    r = new_upload(client, guest[0], **bad)
    assert r.status_code == 400 and r.json()["code"] == "bad_request"


def test_uploads_require_auth(client):
    assert new_upload(client, {}).status_code == 401


def test_get_upload_owner_only(client, guest):
    upload_id = new_upload(client, guest[0], contentType="audio/flac").json()["uploadId"]
    r = client.get(f"/uploads/{upload_id}", headers=guest[0])
    assert r.status_code == 200
    body = r.json()
    assert body["url"] == f"https://s3.test/get/uploads/{upload_id}?exp=300"
    assert body["contentType"] == "audio/flac" and "expiresAt" in body

    other = client.post("/auth/anonymous", json={"deviceId": "device-2", "platform": "android"}).json()["token"]
    for uid in (upload_id, "not-a-uuid", "00000000-0000-0000-0000-000000000000"):
        r = client.get(f"/uploads/{uid}", headers=bearer(other))
        assert r.status_code == 404 and r.json()["code"] == "not_found"


@pytest.mark.skipif(os.environ.get("MINIO_IT") != "1",
                    reason="integration: needs MinIO on localhost:9000 (docker compose up minio minio-init)")
def test_presigned_put_against_minio(monkeypatch):
    from app.settings import get_settings
    from app.storage import Storage
    s = get_settings()
    for k, v in {"s3_endpoint": "http://localhost:9000", "s3_access_key": "minioadmin",
                 "s3_secret_key": "minioadmin", "s3_bucket": "musicdecoded"}.items():
        monkeypatch.setattr(s, k, v)
    storage, data = Storage(), b"ID3 fake mp3 bytes"
    key = f"uploads/it-{hashlib.sha256(data).hexdigest()[:8]}"
    put = httpx.put(storage.presign_put(key, "audio/mpeg", 60), content=data, headers={"Content-Type": "audio/mpeg"})
    assert put.status_code == 200, put.text
    assert httpx.get(storage.presign_get(key, 60)).content == data
