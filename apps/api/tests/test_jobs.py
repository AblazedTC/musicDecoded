import json
import uuid

import pytest

from app.db import SessionLocal
from app.jobs import QUEUE
from app.kv import rate_key
from app.models import Job, Track
from app.settings import get_settings

from conftest import bearer

SHA = "b" * 64
VIDEO = "dQw4w9WgXcQ"


def upload(client, headers, sha=SHA):
    return client.post("/uploads", headers=headers,
                       json={"contentType": "audio/mpeg", "sizeBytes": 10, "sha256": sha}).json()["uploadId"]


def queued(redis_fake):
    return [json.loads(m) for m in redis_fake.lrange(QUEUE, 0, -1)]


def mark_analyzed(track_id):
    with SessionLocal() as db:
        db.get(Track, uuid.UUID(track_id)).analysis_key = f"analysis/{track_id}.json"
        db.commit()


def other_guest(client, device="device-2"):
    return bearer(client.post("/auth/anonymous", json={"deviceId": device, "platform": "android"}).json()["token"])


def test_file_job_is_queued(client, guest, redis_fake):
    headers, _ = guest
    upload_id = upload(client, headers)
    r = client.post("/jobs", headers=headers, json={"source": "file", "uploadId": upload_id, "title": "Song"})
    assert r.status_code == 202
    job = r.json()
    assert job == {"jobId": job["jobId"], "trackId": job["trackId"], "status": "queued", "stage": None,
                   "progress": 0, "error": None}
    assert queued(redis_fake) == [{"jobId": job["jobId"], "trackId": job["trackId"], "source": "file",
                                   "uploadKey": f"uploads/{upload_id}", "videoId": None, "soundcloudUrl": None,
                                   "storeExtractedAudio": False}]
    assert redis_fake.hgetall(f"job:{job['jobId']}") == {"status": "queued", "stage": "", "progress": "0", "error": ""}
    assert 0 < redis_fake.ttl(f"job:{job['jobId']}") <= 86400
    with SessionLocal() as db:
        track = db.get(Track, uuid.UUID(job["trackId"]))
    assert (track.dedupe_key, track.source, track.title) == (f"sha256:{SHA}", "file", "Song")


def test_youtube_job_message(client, guest, redis_fake):
    r = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": VIDEO})
    assert r.status_code == 202
    msg = queued(redis_fake)[0]
    assert (msg["source"], msg["videoId"], msg["uploadKey"]) == ("youtube", VIDEO, None)


def test_same_content_shares_one_track_and_cache_hits_skip_the_queue(client, guest, redis_fake):
    first = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": VIDEO}).json()
    mark_analyzed(first["trackId"])
    other = other_guest(client)
    r = client.post("/jobs", headers=other, json={"source": "youtube", "videoId": VIDEO})
    assert r.status_code == 200
    hit = r.json()
    assert hit["trackId"] == first["trackId"] and hit["jobId"] != first["jobId"]
    assert (hit["status"], hit["progress"]) == ("complete", 100)
    assert len(queued(redis_fake)) == 1  # only the first request was enqueued
    assert client.get(f"/jobs/{hit['jobId']}", headers=other).json()["status"] == "complete"
    assert client.get("/config", headers=other).json()["limits"]["jobsUsedToday"] == 0  # cache hit not counted


def test_byte_identical_uploads_dedupe(client, guest, redis_fake):
    a = client.post("/jobs", headers=guest[0], json={"source": "file", "uploadId": upload(client, guest[0])}).json()
    mark_analyzed(a["trackId"])
    other = other_guest(client)
    b = client.post("/jobs", headers=other, json={"source": "recording", "uploadId": upload(client, other)})
    assert b.status_code == 200 and b.json()["trackId"] == a["trackId"]


def test_repeat_request_while_in_flight_returns_same_job(client, guest, redis_fake):
    a = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": VIDEO}).json()
    b = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": VIDEO})
    assert b.status_code == 202 and b.json()["jobId"] == a["jobId"]
    assert len(queued(redis_fake)) == 1


def test_daily_limit_is_429(client, guest, redis_fake, monkeypatch):
    monkeypatch.setattr(get_settings(), "limit_guest", 2)
    for i in range(2):
        r = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": f"video{i:06d}"})
        assert r.status_code == 202
    r = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": "video999999"})
    assert r.status_code == 429 and r.json()["code"] == "rate_limited"
    assert redis_fake.get(rate_key(guest[1]["userId"])) == "2"
    assert len(queued(redis_fake)) == 2


@pytest.mark.parametrize("body", [
    {"source": "file"}, {"source": "recording"}, {"source": "youtube"},
    {"source": "youtube", "videoId": "short"}, {"source": "vimeo"}, {},
])
def test_source_validation_is_400(client, guest, body):
    r = client.post("/jobs", headers=guest[0], json=body)
    assert r.status_code == 400 and r.json()["code"] == "bad_request"


def test_unknown_or_foreign_upload_is_404(client, guest):
    foreign = upload(client, other_guest(client))
    for upload_id in (foreign, str(uuid.uuid4()), "nope"):
        r = client.post("/jobs", headers=guest[0], json={"source": "file", "uploadId": upload_id})
        assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_soundcloud_disabled_is_403(client, guest):
    r = client.post("/jobs", headers=guest[0], json={"source": "soundcloud", "soundcloudUrl": "https://soundcloud.com/a/b"})
    assert r.status_code == 403 and r.json()["code"] == "feature_disabled"


def test_soundcloud_enabled(client, guest, redis_fake, monkeypatch):
    monkeypatch.setattr(get_settings(), "soundcloud_enabled", True)
    for url in ("https://evil.example/a/b", "file:///etc/passwd", "https://soundcloud.com/", None):
        r = client.post("/jobs", headers=guest[0], json={"source": "soundcloud", "soundcloudUrl": url})
        assert r.status_code == 400, url
    r = client.post("/jobs", headers=guest[0], json={"source": "soundcloud", "soundcloudUrl": "https://soundcloud.com/a/b/"})
    assert r.status_code == 202
    assert queued(redis_fake)[0]["soundcloudUrl"] == "https://soundcloud.com/a/b"


def test_get_job_reads_live_redis_then_postgres(client, guest, redis_fake):
    job = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": VIDEO}).json()
    redis_fake.hset(f"job:{job['jobId']}", mapping={"status": "processing", "stage": "detecting_chords",
                                                      "progress": "40"})
    body = client.get(f"/jobs/{job['jobId']}", headers=guest[0]).json()
    assert (body["status"], body["stage"], body["progress"], body["error"]) == \
        ("processing", "detecting_chords", 40, None)

    redis_fake.delete(f"job:{job['jobId']}")  # expired → Postgres
    with SessionLocal() as db:
        db.get(Job, uuid.UUID(job["jobId"])).status = "failed"
        db.get(Job, uuid.UUID(job["jobId"])).error = "boom"
        db.commit()
    body = client.get(f"/jobs/{job['jobId']}", headers=guest[0]).json()
    assert (body["status"], body["error"]) == ("failed", "boom")


def test_get_job_of_another_user_is_404(client, guest):
    job = client.post("/jobs", headers=guest[0], json={"source": "youtube", "videoId": VIDEO}).json()
    for headers, job_id in ((other_guest(client), job["jobId"]), (guest[0], str(uuid.uuid4())), (guest[0], "x")):
        r = client.get(f"/jobs/{job_id}", headers=headers)
        assert r.status_code == 404 and r.json()["code"] == "not_found"
