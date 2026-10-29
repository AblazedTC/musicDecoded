"""T-402: GET/PUT /sync/library (last-write-wins on updatedAt, tombstones) and sync-based downloads."""
import uuid

from app import sync

from conftest import bearer
from test_tracks import set_track

T0, T1, T2 = "2026-10-01T10:00:00Z", "2026-10-02T10:00:00Z", "2026-10-03T10:00:00Z"


def account(client, email="a@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    return bearer(r.json()["token"])


def login(client, email="a@example.com"):
    return bearer(client.post("/auth/login", json={"email": email, "password": "password123"}).json()["token"])


def track(client, headers, video="dQw4w9WgXcQ"):
    return client.post("/jobs", headers=headers, json={"source": "youtube", "videoId": video}).json()["trackId"]


def item(track_id, updated=T1, **fields):
    return {"trackId": track_id, "title": "Song", "source": "youtube", "createdAt": T0, "updatedAt": updated,
            "deleted": False, **fields}


def push(client, headers, *items):
    return client.put("/sync/library", headers=headers, json={"items": list(items)})


def pull(client, headers, since=None):
    r = client.get("/sync/library", headers=headers, params={"since": since} if since is not None else {})
    assert r.status_code == 200, r.text
    return r.json()


def test_guest_gets_account_required(client, guest):
    for r in (client.get("/sync/library", headers=guest[0]), push(client, guest[0], item(str(uuid.uuid4())))):
        assert r.status_code == 403 and r.json()["code"] == "account_required"
    assert client.get("/sync/library").status_code == 401


def test_full_pull_returns_contract_shape(client):
    a = account(client)
    tid = track(client, a)
    upload_id = client.post("/uploads", headers=a, json={"contentType": "audio/mpeg", "sizeBytes": 10,
                                                         "sha256": "c" * 64}).json()["uploadId"]
    sent = item(tid, artist="Rick", sourceRef="dQw4w9WgXcQ", durationSec=212.5, thumbnailUrl="https://i.test/t.jpg",
                audioUploadId=upload_id, favorite=True, lastPlayedAt=T1)
    assert push(client, a, sent).status_code == 200
    body = pull(client, a)
    assert body["items"] == [{**sent, "createdAt": T0, "updatedAt": T1}]
    assert body["cursor"]
    assert pull(client, a) == body  # full pull is repeatable


def test_incremental_pull_and_pagination(client, monkeypatch):
    a = account(client)
    tids = [track(client, a, video=f"video{i:06d}") for i in range(3)]
    push(client, a, *[item(t) for t in tids])
    cursor = pull(client, a)["cursor"]
    assert pull(client, a, cursor) == {"items": [], "cursor": cursor}

    push(client, a, item(tids[1], updated=T2, favorite=True))
    changed = pull(client, a, cursor)
    assert [i["trackId"] for i in changed["items"]] == [tids[1]] and changed["items"][0]["favorite"] is True

    monkeypatch.setattr(sync, "PAGE_SIZE", 2)
    first = pull(client, a)
    second = pull(client, a, first["cursor"])
    assert len(first["items"]) == 2 and len(second["items"]) == 1
    assert {i["trackId"] for i in first["items"] + second["items"]} == set(tids)
    assert second["items"][0]["trackId"] == tids[1]  # most recently changed comes last
    assert pull(client, a, second["cursor"])["items"] == []


def test_bad_cursor_is_400(client):
    a = account(client)
    for bad in ("not-base64!", "Zm9v"):
        r = client.get("/sync/library", headers=a, params={"since": bad})
        assert r.status_code == 400 and r.json()["code"] == "bad_request"


def test_last_write_wins(client):
    a = account(client)
    tid = track(client, a)
    push(client, a, item(tid, updated=T1, title="v1"))
    older = push(client, a, item(tid, updated=T0, title="stale")).json()["items"]
    assert older[0]["title"] == "v1" and older[0]["updatedAt"] == T1  # server copy returned
    same = push(client, a, item(tid, updated=T1, title="tie")).json()["items"]
    assert same[0]["title"] == "v1"  # ties keep the server copy
    newer = push(client, a, item(tid, updated=T2, title="v2")).json()["items"]
    assert newer[0]["title"] == "v2" and newer[0]["updatedAt"] == T2
    assert pull(client, a)["items"][0]["title"] == "v2"


def test_last_write_wins_offsets(client):
    a = account(client)
    tid = track(client, a)
    push(client, a, item(tid, updated=T1, title="v1"))
    # 12:00+02:00 == 10:00Z == T1: a tie, server keeps v1. 12:01+02:00 is newer.
    assert push(client, a, item(tid, updated="2026-10-02T12:00:00+02:00", title="x")).json()["items"][0]["title"] == "v1"
    r = push(client, a, item(tid, updated="2026-10-02T12:01:00+02:00", title="v2")).json()["items"][0]
    assert r["title"] == "v2" and r["updatedAt"] == "2026-10-02T10:01:00Z"


def test_tombstones_sync(client):
    a = account(client)
    tid = track(client, a)
    push(client, a, item(tid))
    cursor = pull(client, a)["cursor"]
    push(client, a, item(tid, updated=T2, deleted=True))
    changed = pull(client, a, cursor)["items"]
    assert len(changed) == 1 and changed[0]["deleted"] is True
    # An older non-deleted write can't resurrect it.
    assert push(client, a, item(tid, updated=T1)).json()["items"][0]["deleted"] is True


def test_push_validation(client):
    a, b = account(client), account(client, "b@example.com")
    tid = track(client, a)
    for bad in (item(str(uuid.uuid4())), item("nope")):
        r = push(client, a, bad)
        assert r.status_code == 400 and r.json()["code"] == "bad_request"
    b_upload = client.post("/uploads", headers=b, json={"contentType": "audio/mpeg", "sizeBytes": 10,
                                                        "sha256": "c" * 64}).json()["uploadId"]
    r = push(client, a, item(tid), item(tid, updated=T2, audioUploadId=b_upload))
    assert r.status_code == 400 and r.json()["code"] == "bad_request"
    assert pull(client, a)["items"] == []  # a rejected push writes nothing
    assert push(client, a, item(tid, source="vinyl")).status_code == 400


def test_cross_user_isolation(client):
    a, b = account(client), account(client, "b@example.com")
    tid = track(client, a)
    push(client, a, item(tid, title="mine"))
    assert pull(client, b)["items"] == []
    push(client, b, item(tid, updated=T0, title="theirs"))  # same track, separate row per user
    assert pull(client, a)["items"][0]["title"] == "mine"
    assert pull(client, b)["items"][0]["title"] == "theirs"


def test_library_row_grants_downloads(client, guest):
    tid = track(client, guest[0])  # someone else's job created the track
    set_track(tid, analysis_key=f"analysis/{tid}.json")
    a = account(client)
    assert client.get(f"/tracks/{tid}/analysis", headers=a).status_code == 404
    push(client, a, item(tid))
    assert client.get(f"/tracks/{tid}/analysis", headers=a).status_code == 200
    push(client, a, item(tid, updated=T2, deleted=True))
    assert client.get(f"/tracks/{tid}/analysis", headers=a).status_code == 404  # tombstoned: no access


def test_two_devices_converge(client):
    """Device A decodes a recording and syncs; device B (same account) pulls, downloads, edits;
    A pushes a stale edit made offline; both end with the same state."""
    a = account(client)
    upload_id = client.post("/uploads", headers=a, json={"contentType": "audio/mpeg", "sizeBytes": 10,
                                                         "sha256": "d" * 64}).json()["uploadId"]
    tid = client.post("/jobs", headers=a, json={"source": "file", "uploadId": upload_id}).json()["trackId"]
    set_track(tid, analysis_key=f"analysis/{tid}.json")
    push(client, a, item(tid, source="file", audioUploadId=upload_id))
    a_cursor = pull(client, a)["cursor"]

    b = login(client)
    b_pull = pull(client, b)
    assert [i["trackId"] for i in b_pull["items"]] == [tid]
    assert client.get(f"/tracks/{tid}/analysis", headers=b).status_code == 200
    assert client.get(f"/uploads/{b_pull['items'][0]['audioUploadId']}", headers=b).status_code == 200

    push(client, b, item(tid, updated=T2, source="file", audioUploadId=upload_id, favorite=True))
    stale = push(client, a, item(tid, updated=T1, source="file", audioUploadId=upload_id, title="Renamed")).json()
    assert stale["items"][0]["favorite"] is True and stale["items"][0]["title"] == "Song"  # B's newer write won

    a_state = pull(client, a, a_cursor)["items"]
    b_state = pull(client, b, b_pull["cursor"])["items"]
    assert a_state == b_state == pull(client, a)["items"]
    assert a_state[0]["favorite"] is True
