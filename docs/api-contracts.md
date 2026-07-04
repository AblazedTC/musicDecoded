# API Contracts

The Android app communicates exclusively with the Spring Boot API over HTTPS/REST.

---

## Auth

### `POST /auth/signup`

Create a new user account.

**Request**

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response `201`**

```json
{
  "token": "eyJhbGciOiJIUzI1NiJ9..."
}
```

---

### `POST /auth/login`

Authenticate and receive a JWT.

**Request**

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response `200`**

```json
{
  "token": "eyJhbGciOiJIUzI1NiJ9..."
}
```

---

## Tracks

### `POST /tracks/youtube`

Submit a YouTube link. Creates or reuses a track and queues analysis if needed.

**Request**

```json
{
  "youtubeUrl": "https://www.youtube.com/watch?v=abc123"
}
```

**Response `202` — new job created**

```json
{
  "trackId": "track_123",
  "libraryItemId": "lib_456",
  "analysisStatus": "queued",
  "jobId": "job_789"
}
```

**Response `200` — analysis already exists**

```json
{
  "trackId": "track_123",
  "libraryItemId": "lib_456",
  "analysisStatus": "complete"
}
```

---

### `GET /tracks/{trackId}`

Get track metadata.

**Response `200`**

```json
{
  "id": "track_123",
  "title": "Song Title",
  "channelName": "Artist Name",
  "durationSeconds": 214,
  "thumbnailUrl": "https://...",
  "providerVideoId": "abc123"
}
```

---

### `GET /tracks/{trackId}/analysis`

Get analysis metadata and artifact URLs for a track.

> Requires the track to be in the current user's library.

**Response `200`**

```json
{
  "trackId": "track_123",
  "bpm": 120.4,
  "keySignature": "F# minor",
  "timeSignature": "4/4",
  "artifacts": {
    "chords": "https://storage/.../chords.json",
    "beats": "https://storage/.../beats.json",
    "grid": "https://storage/.../grid.json",
    "waveform": "https://storage/.../waveform.json"
  }
}
```

---

## Library

### `GET /library`

Get the current user's saved tracks.

**Response `200`**

```json
[
  {
    "libraryItemId": "lib_456",
    "track": {
      "id": "track_123",
      "title": "Song Title",
      "thumbnailUrl": "https://...",
      "durationSeconds": 214
    },
    "analysisStatus": "complete",
    "lastPlayedAt": "2026-07-04T12:00:00Z"
  }
]
```

---

### `DELETE /library/{libraryItemId}`

Remove a track from the user's library.

**Response `204`** — no content

---

## Jobs

### `GET /jobs/{jobId}`

Poll the status of an analysis job.

**Response `200`**

```json
{
  "jobId": "job_789",
  "status": "processing",
  "stage": "detecting_chords",
  "progress": 72
}
```

**Possible `status` values:** `queued`, `processing`, `complete`, `failed`

**Possible `stage` values:**

```
queued
resolving_youtube_media
downloading_temp_media
extracting_audio
normalizing_audio
generating_waveform
detecting_bpm
detecting_beats
detecting_downbeats
detecting_key
detecting_chords
building_grid
generating_explanation
saving_results
complete
failed
```

---

## Queue message format

When Spring Boot pushes a job to the queue, the message contains:

```json
{
  "jobId": "job_789",
  "trackId": "track_123",
  "youtubeUrl": "https://www.youtube.com/watch?v=abc123",
  "videoId": "abc123"
}
```

---

## Authentication

All endpoints except `/auth/signup` and `/auth/login` require a JWT in the `Authorization` header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9...
```

> **Android note:** When using the Android emulator, `localhost` points to the emulator itself, not your machine. Use `http://10.0.2.2:8080` to reach a locally running API.
