"""T-401: signup/login with a guest bearer moves that guest's jobs and uploads to the account."""
from conftest import bearer

CREDS = {"email": "a@example.com", "password": "password123"}
VIDEO = "dQw4w9WgXcQ"


def guest_with_data(client, device="device-1"):
    """A guest with one upload and one job; returns (headers, uploadId, jobId)."""
    headers = bearer(client.post("/auth/anonymous", json={"deviceId": device, "platform": "ios"}).json()["token"])
    upload_id = client.post("/uploads", headers=headers,
                            json={"contentType": "audio/mpeg", "sizeBytes": 10, "sha256": "c" * 64}).json()["uploadId"]
    job_id = client.post("/jobs", headers=headers, json={"source": "youtube", "videoId": VIDEO}).json()["jobId"]
    return headers, upload_id, job_id


def owns(client, headers, upload_id, job_id) -> tuple[bool, bool]:
    return (client.get(f"/uploads/{upload_id}", headers=headers).status_code == 200,
            client.get(f"/jobs/{job_id}", headers=headers).status_code == 200)


def test_signup_with_guest_token_moves_data(client):
    guest, upload_id, job_id = guest_with_data(client)
    r = client.post("/auth/signup", json=CREDS, headers=guest)
    assert r.status_code == 201 and r.json()["kind"] == "account"
    account = bearer(r.json()["token"])
    assert owns(client, account, upload_id, job_id) == (True, True)
    assert owns(client, guest, upload_id, job_id) == (False, False)
    # The guest row survives: the device re-auths as the same (now empty) guest.
    again = client.post("/auth/anonymous", json={"deviceId": "device-1", "platform": "ios"}).json()
    assert again["kind"] == "guest" and client.get("/_test/me", headers=guest).status_code == 200


def test_login_merges_guest_data(client):
    existing = bearer(client.post("/auth/signup", json=CREDS).json()["token"])
    guest, upload_id, job_id = guest_with_data(client, device="device-2")
    r = client.post("/auth/login", json=CREDS, headers=guest)
    assert r.status_code == 200
    assert owns(client, existing, upload_id, job_id) == (True, True)
    assert owns(client, guest, upload_id, job_id) == (False, False)


def test_without_guest_token_nothing_moves(client):
    guest, upload_id, job_id = guest_with_data(client)
    account = bearer(client.post("/auth/signup", json=CREDS).json()["token"])
    assert client.post("/auth/login", json=CREDS).status_code == 200
    assert owns(client, account, upload_id, job_id) == (False, False)
    assert owns(client, guest, upload_id, job_id) == (True, True)


def test_account_or_invalid_bearer_is_ignored(client):
    other = {"email": "b@example.com", "password": "password123"}
    other_headers = bearer(client.post("/auth/signup", json=other).json()["token"])
    upload_id = client.post("/uploads", headers=other_headers,
                            json={"contentType": "audio/mpeg", "sizeBytes": 10, "sha256": "d" * 64}).json()["uploadId"]
    r = client.post("/auth/signup", json=CREDS, headers=other_headers)  # an account token is not a guest link
    assert r.status_code == 201
    assert client.get(f"/uploads/{upload_id}", headers=bearer(r.json()["token"])).status_code == 404
    assert client.get(f"/uploads/{upload_id}", headers=other_headers).status_code == 200
    assert client.post("/auth/login", json=CREDS, headers=bearer("not-a-jwt")).status_code == 200


def test_duplicate_email_is_409_and_keeps_guest_data(client):
    client.post("/auth/signup", json=CREDS)
    guest, upload_id, job_id = guest_with_data(client)
    r = client.post("/auth/signup", json={**CREDS, "email": "A@example.com"}, headers=guest)
    assert r.status_code == 409 and r.json()["code"] == "conflict"
    assert owns(client, guest, upload_id, job_id) == (True, True)


def test_wrong_password_does_not_merge(client):
    client.post("/auth/signup", json=CREDS)
    guest, upload_id, job_id = guest_with_data(client)
    assert client.post("/auth/login", json={**CREDS, "password": "wrong-password"}, headers=guest).status_code == 401
    assert owns(client, guest, upload_id, job_id) == (True, True)
