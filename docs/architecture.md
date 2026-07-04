# Architecture

## Overview

MusicDecoded lets users submit a YouTube link and receive deep music analysis (beats, downbeats, chords, grid) synced to YouTube playback. The system does **not** permanently store downloaded audio or video — only metadata and analysis artifacts.

---

## System Diagram

```
Kotlin Android App
      |
      | HTTPS / REST
      v
Java Spring Boot API
      |
      |── PostgreSQL      (permanent data)
      |── Redis           (fast temporary state / cache)
      |── Object Storage  (large JSON artifacts)
      └── Queue           (async job handoff)
              |
              v
      Python Worker
              |
              |── youtube-video-downloader-api
              |── FFmpeg
              └── Beat / Downbeat / Chord Models
                          |
                          v
              Analysis Artifacts (Object Storage)
```

**Important rule:** The Android app only ever talks to Spring Boot. It never calls the worker, database, object storage, or downloader API directly. All secrets and access control live in the API.

---

## Why This Architecture

Audio analysis is slow. A single job involves:

- Downloading temporary media
- Extracting and normalizing audio
- Generating spectrograms / chroma features
- Running beat, downbeat, and chord models
- Post-processing timestamps
- Saving artifacts

That is too long for a normal HTTP request. So instead of making the client wait, the system uses an async job pattern:

```
User submits link
  → API creates job immediately
  → Worker processes in background
  → App polls job status
  → App loads result when complete
```

This keeps the API responsive and lets the worker scale independently.

---

## Main Technologies

### Backend

- Java, Spring Boot, Spring Security
- JWT authentication
- PostgreSQL, Redis
- REST API
- Docker

### Mobile

- Kotlin, Jetpack Compose
- Retrofit / Ktor HTTP client
- Coroutines, ViewModels

### Worker / ML

- Python, FFmpeg, PyTorch
- Beat Transformer (beat/downbeat detection)
- CNN-LSTM (chord recognition)
- youtube-video-downloader-api

### Infrastructure

- Docker Compose (local)
- PostgreSQL, Redis, MinIO (local) or S3/R2 (production)
- Optional: Hugging Face / Modal / Replicate for hosted model inference

---

## Repo Structure

```
musicdecoded/
  apps/
    api/          # Java Spring Boot backend
    mobile/       # Kotlin Android app
    worker/       # Python worker
  services/
    audio-engine/ # optional C++ module later
  infra/
    docker-compose.yml
    terraform/
  docs/
    architecture.md
    api-contracts.md
    data-model.md
    user-flows.md
  README.md
```

### Spring Boot package structure

```
apps/api/src/main/java/com/musicdecoded/api/
  auth/
  users/
  tracks/
  library/
  jobs/
  analysis/
  queue/
  storage/
  common/
```

### Python worker structure

```
apps/worker/src/musicdecoded_worker/
  providers/
  media/
  analysis/
  models/
  storage/
  queue/
  jobs/
  db/
  common/
```

### Android structure

```
apps/mobile/app/src/main/java/com/musicdecoded/mobile/
  core/
  auth/
  library/
  tracks/
  jobs/
  analysis/
  player/
  model/
```

---

## Hosting Plan

### Phase 1 — Local only

Run everything locally until the core flow works.

- Spring Boot (local)
- Python worker (local)
- PostgreSQL, Redis, MinIO (Docker)
- Android emulator

### Phase 2 — Hosted managed services

Keep API/worker local, use online infra for real connection string testing.

- Neon (PostgreSQL)
- Upstash (Redis)
- Cloudflare R2 (object storage)

### Phase 3 — Host the API

Deploy Spring Boot to Railway, Render, or Fly.io. Android now calls a real HTTPS backend.

### Phase 4 — Host the worker

Deploy Python worker to Railway, Render, Fly.io, or Modal (if model-heavy).

### Phase 5 — AWS (when ready)

- Spring Boot → ECS Fargate
- Python Worker → ECS Fargate
- PostgreSQL → RDS
- Queue → SQS
- Object Storage → S3
- Redis → ElastiCache
- Models → SageMaker / ECS GPU

Start with cheaper platforms (Railway, Neon, Upstash, R2). Migrate to AWS when the app is worth the setup cost or you specifically want AWS experience.

---

## Security Rules

- The Android app only talks to Spring Boot — never directly to the worker, DB, or storage
- Spring Boot verifies JWTs on all protected routes
- Spring Boot checks `user_library` before returning any analysis data — users can only see tracks in their own library
- The downloader API key lives only in the Python worker environment
- The database is not exposed publicly
- Object storage is private; use signed URLs for client access
- Rate limits protect the worker and downloader API from abuse
