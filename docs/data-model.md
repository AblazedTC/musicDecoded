# Data Model

## Database (PostgreSQL)

### `users`

Stores user accounts.

| Column        | Type          | Notes                           |
| ------------- | ------------- | ------------------------------- |
| id            | UUID / BIGINT | Primary key                     |
| email         | VARCHAR       | Unique                          |
| password_hash | VARCHAR       | BCrypt hashed, never plain text |
| display_name  | VARCHAR       |                                 |
| created_at    | TIMESTAMP     |                                 |

---

### `tracks`

Global track/video table. A track represents a YouTube video and is **not** owned by one user.

| Column            | Type          | Notes            |
| ----------------- | ------------- | ---------------- |
| id                | UUID / BIGINT | Primary key      |
| source_type       | VARCHAR       | `youtube`        |
| provider          | VARCHAR       | `youtube`        |
| provider_video_id | VARCHAR       | YouTube video ID |
| title             | VARCHAR       |                  |
| channel_name      | VARCHAR       |                  |
| duration_seconds  | INT           |                  |
| thumbnail_url     | VARCHAR       |                  |
| playback_type     | VARCHAR       | `youtube_embed`  |
| created_at        | TIMESTAMP     |                  |

**Why global?** If 10 users submit the same YouTube link, the track is only processed once. All 10 users point to the same track record.

---

### `user_library`

Connects users to tracks. This is how each user has their own collection.

| Column         | Type          | Notes       |
| -------------- | ------------- | ----------- |
| id             | UUID / BIGINT | Primary key |
| user_id        | FK → users    |             |
| track_id       | FK → tracks   |             |
| created_at     | TIMESTAMP     |             |
| last_played_at | TIMESTAMP     | Nullable    |

```
tracks are global
user_library is per-user

User A saves Track X  →  user_library row (user=A, track=X)
User B saves Track X  →  user_library row (user=B, track=X)
Both point to the same global Track X
```

---

### `analysis_jobs`

Tracks async processing state.

| Column               | Type          | Notes                                        |
| -------------------- | ------------- | -------------------------------------------- |
| id                   | UUID / BIGINT | Primary key                                  |
| track_id             | FK → tracks   |                                              |
| requested_by_user_id | FK → users    |                                              |
| status               | VARCHAR       | `queued`, `processing`, `complete`, `failed` |
| stage                | VARCHAR       | Current worker stage                         |
| progress             | INT           | 0–100                                        |
| error_message        | TEXT          | Nullable                                     |
| created_at           | TIMESTAMP     |                                              |
| started_at           | TIMESTAMP     | Nullable                                     |
| completed_at         | TIMESTAMP     | Nullable                                     |

---

### `track_analysis`

Stores final analysis metadata and artifact keys (not the large arrays themselves).

| Column            | Type          | Notes                  |
| ----------------- | ------------- | ---------------------- |
| id                | UUID / BIGINT | Primary key            |
| track_id          | FK → tracks   |                        |
| bpm               | FLOAT         |                        |
| key_signature     | VARCHAR       | e.g. `F# minor`        |
| time_signature    | VARCHAR       | e.g. `4/4`             |
| chords_json_key   | VARCHAR       | Path in object storage |
| beats_json_key    | VARCHAR       | Path in object storage |
| grid_json_key     | VARCHAR       | Path in object storage |
| waveform_json_key | VARCHAR       | Path in object storage |
| lyrics_json_key   | VARCHAR       | Nullable               |
| sections_json_key | VARCHAR       | Nullable               |
| ai_summary        | TEXT          | Nullable               |
| created_at        | TIMESTAMP     |                        |

Large arrays are stored in object storage. PostgreSQL only stores the path/key to retrieve them.

---

## Object Storage (MinIO / S3 / R2)

Used for large static JSON artifacts that don't belong in a relational DB.

### Artifact paths

```
tracks/{trackId}/analysis/chords.json
tracks/{trackId}/analysis/beats.json
tracks/{trackId}/analysis/grid.json
tracks/{trackId}/analysis/waveform.json
tracks/{trackId}/analysis/sections.json
tracks/{trackId}/analysis/lyrics.json
```

### `chords.json`

```json
[
  { "start": 0.0, "end": 2.4, "chord": "F#m7", "confidence": 0.91 },
  { "start": 2.4, "end": 4.8, "chord": "Dmaj7", "confidence": 0.87 }
]
```

### `beats.json`

```json
[
  { "time": 0.0, "bar": 1, "beat": 1 },
  { "time": 0.65, "bar": 1, "beat": 2 }
]
```

### `grid.json`

```json
[
  {
    "bar": 1,
    "beats": [
      { "beat": 1, "time": 0.0, "chord": "F#m7" },
      { "beat": 2, "time": 0.65, "chord": "F#m7" },
      { "beat": 3, "time": 1.3, "chord": "Dmaj7" },
      { "beat": 4, "time": 1.95, "chord": "Dmaj7" }
    ]
  }
]
```

---

## Redis

Used for fast temporary state — primarily job progress polling.

### Key patterns

```
job:{jobId}:status              = processing
job:{jobId}:stage               = detecting_chords
job:{jobId}:progress            = 72
youtube:{videoId}:trackId       = track_123
rate:user:{userId}:submissions  = 3
```

**Why Redis for job progress?**
The Android app polls `GET /jobs/{jobId}` every few seconds during processing. Serving those reads from Redis avoids hammering PostgreSQL with constant status-check queries. PostgreSQL stores the durable job record; Redis stores the fast-changing progress.

---

## What Is and Isn't Stored Permanently

| Stored                                          | Not stored               |
| ----------------------------------------------- | ------------------------ |
| Video ID, title, channel, duration, thumbnail   | Downloaded YouTube audio |
| BPM, key, time signature                        | Downloaded YouTube video |
| Chords, beats, grid (as JSON in object storage) | Normalized WAV file      |
| Waveform, lyrics, sections                      |                          |
| AI/theory summary                               |                          |

Temporary files (downloaded media, normalized WAV) only exist on the worker during job processing and are deleted when the job completes.
