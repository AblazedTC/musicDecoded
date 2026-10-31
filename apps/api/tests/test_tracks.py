import uuid

from app.db import SessionLocal
from app.models import Track
from app.settings import get_settings

from conftest import bearer

VIDEO = "dQw4w9WgXcQ"


def start_job(client, headers):
    return client.post("/jobs", headers=headers, json={"source": "youtube", "videoId": VIDEO}).json()["trackId"]


def set_track(track_id, **fields):
    with SessionLocal() as db:
        track = db.get(Track, uuid.UUID(track_id))
        for k, v in fields.items():
            setattr(track, k, v)
        db.commit()


def other_guest(client):
    return bearer(client.post("/auth/anonymous", json={"deviceId": "device-2", "platform": "ios"}).json()["token"])


def test_analysis_404_until_ready_then_presigned(client, guest):
    track_id = start_job(client, guest[0])
    r = client.get(f"/tracks/{track_id}/analysis", headers=guest[0])
    assert r.status_code == 404 and r.json()["code"] == "not_found"
    set_track(track_id, analysis_key=f"analysis/{track_id}.json")
    r = client.get(f"/tracks/{track_id}/analysis", headers=guest[0])
    assert r.status_code == 200
    assert r.json() == {"url": f"https://s3.test/get/analysis/{track_id}.json?exp=300", "schemaVersion": 1}


def test_analysis_needs_a_link_to_the_track(client, guest):
    track_id = start_job(client, guest[0])
    set_track(track_id, analysis_key=f"analysis/{track_id}.json")
    other = other_guest(client)
    for tid in (track_id, str(uuid.uuid4()), "nope"):
        assert client.get(f"/tracks/{tid}/analysis", headers=other).status_code == 404
    # Once the other user requests the same video (a cache hit), they're linked.
    client.post("/jobs", headers=other, json={"source": "youtube", "videoId": VIDEO})
    assert client.get(f"/tracks/{track_id}/analysis", headers=other).status_code == 200


def test_audio_is_feature_flagged(client, guest, monkeypatch):
    track_id = start_job(client, guest[0])
    r = client.get(f"/tracks/{track_id}/audio", headers=guest[0])
    assert r.status_code == 403 and r.json()["code"] == "feature_disabled"

    monkeypatch.setattr(get_settings(), "offline_stream_audio", True)
    assert client.get(f"/tracks/{track_id}/audio", headers=guest[0]).status_code == 404
    set_track(track_id, audio_key=f"extracted-audio/{track_id}.m4a")
    r = client.get(f"/tracks/{track_id}/audio", headers=guest[0])
    assert r.status_code == 200
    assert r.json() == {"url": f"https://s3.test/get/extracted-audio/{track_id}.m4a?exp=300", "contentType": "audio/mp4"}
    assert client.get(f"/tracks/{track_id}/audio", headers=other_guest(client)).status_code == 404
