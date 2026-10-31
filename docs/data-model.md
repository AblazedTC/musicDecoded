# Data Model

Owned by the `contracts` lane. Wire formats are defined in `contracts/` and this doc must agree with them.

## 1. On-device (iOS and Android, identical)

```
<app support dir>/audio/{trackId}.{ext}      user's file/recording, or flag-gated stream audio
<app support dir>/analysis/{trackId}.json    analysis v1 (contracts/analysis.schema.json), verbatim
```

**LibraryItem** (Room entity / SwiftData model) mirrors `LibraryItem` in `contracts/openapi.yaml`, plus local-only fields:

| Field | Type | Notes |
|---|---|---|
| trackId | String (PK) | from the server job response |
| title, artist | String / String? | |
| source | enum | file · recording · youtube · soundcloud |
| sourceRef | String? | videoId / SoundCloud URL |
| durationSec | Double? | |
| thumbnailUrl | String? | |
| audioUploadId | String? | file/recording only; enables sync download |
| favorite | Bool | Library "Favorites" filter; synced |
| key, bpm | String? / Double? | copied from the analysis file so lists can show "3:22 · B minor · 140 BPM" and search by key without decoding JSON |
| lastPlayedAt | Instant? | null until first played |
| createdAt, updatedAt | Instant | `updatedAt` drives last-write-wins |
| deleted | Bool | tombstone |
| *local:* status | enum | analyzing · ready · failed |
| *local:* jobId | String? | while analyzing |
| *local:* localAudioPath | String? | null if streaming-only |
| *local:* syncedAt | Instant? | |

## 2. Server (PostgreSQL)

| Table | Key columns | Notes |
|---|---|---|
| `users` | id UUID, kind (guest/account), plan (free/pro, default free), plan_expires_at (nullable), device_id (unique, nullable), email (unique, nullable), password_hash (nullable), created_at | A guest becomes an account when its jobs/uploads are reassigned |
| `uploads` | id, user_id, sha256, content_type, size_bytes, object_key, created_at | object_key = `uploads/{id}` |
| `tracks` | id, source, dedupe_key (unique), title, duration_sec, thumbnail_url, analysis_key (nullable), audio_key (nullable), analyzer_version, created_at | Global. dedupe_key = `sha256:<hex>` / `yt:<videoId>` / `sc:<url>` |
| `jobs` | id, track_id, user_id, status, stage, progress, error, created_at, started_at, completed_at | Durable record; live progress is mirrored in Redis |
| `library_items` | user_id + track_id (PK), title, artist, source, source_ref, duration_sec, thumbnail_url, audio_upload_id, favorite, last_played_at, created_at, updated_at, deleted | **Accounts only**; the server copy used for sync |

## 3. Object storage

```
uploads/{uploadId}                  user audio (files/recordings)
analysis/{trackId}.json             analysis v1
extracted-audio/{trackId}.m4a       streaming-source audio, ONLY when OFFLINE_STREAM_AUDIO=true
```

## 4. Redis

```
jobs:analysis                 LIST   FIFO queue (API RPUSH, worker BLPOP)
job:{jobId}                   HASH   status, stage, progress, error   (TTL 1 day)
yt:search:{sha1(q|pageToken)} STRING cached search response           (TTL 1 hour)
rate:{userId}:{yyyymmdd}      INT    job count                        (TTL 2 days)
```

### Queue message (`jobs:analysis`)
```json
{
  "jobId": "uuid",
  "trackId": "uuid",
  "source": "file | recording | youtube | soundcloud",
  "uploadKey": "uploads/<id> | null",
  "videoId": "string | null",
  "soundcloudUrl": "string | null",
  "storeExtractedAudio": false
}
```
