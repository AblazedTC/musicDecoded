# Architecture

The product-level architecture, decisions and risks live in [`PLAN.md`](../PLAN.md) §5–7. This doc covers component internals and hosting.

## Components

| Component | Responsibility | Talks to |
|---|---|---|
| iOS app (`apps/ios`) | UI, local library, playback, recording, sync client | API only (+ presigned URLs) |
| Android app (`apps/android`) | same as iOS | API only (+ presigned URLs) |
| API (`apps/api`) | Auth (guest/account), uploads, jobs, analysis/audio download, YouTube search proxy, sync, flags | Postgres, Redis, object storage, YouTube Data API |
| Worker (`apps/worker`) | Fetch media → normalize → beats/chords/key/bpm/timesig/spectrogram → analysis file | Redis (queue), Postgres (job status), object storage |

## Request lifecycle: transcribe a local file

```
App                      API                          Redis      Worker               Storage
 | POST /uploads ------->|                              |           |                    |
 |<-- putUrl ------------|                              |           |                    |
 | PUT bytes ---------------------------------------------------------------------------->| uploads/{id}
 | POST /jobs ---------->| dedupe by sha256             |           |                    |
 |                       |-- LPUSH jobs:analysis ------>|           |                    |
 |<-- 202 {jobId} -------|                              |-- BLPOP ->|                    |
 | GET /jobs/{id} (poll)>| reads job:{id}:* from Redis  |<- stage --|-- get upload ----->|
 |                       |                              |           |-- put analysis --->| analysis/{trackId}.json
 |                       |                              |<- done ---|                    |
 | GET /tracks/{id}/analysis -> presigned GET url       |           |                    |
 | GET url ----------------------------------------------------------------------------->|
 | save analysis/{trackId}.json locally, mark item ready, open player
```

YouTube/SoundCloud are the same, except there is no upload step: `POST /jobs {source: youtube, videoId}`, and the worker fetches the media itself.

## Security rules
- All secrets live only in API/worker env vars. Apps hold only their own JWT and deviceId (Keychain/Keystore).
- Presigned URLs expire within 15 minutes (PUT) or 5 minutes (GET).
- Every lookup is scoped to the caller's userId. Analysis files are shared globally, but you can only get one for a track in your own jobs or library.
- Rate limits are per userId (see T-104).

## Hosting plan

| Phase | What |
|---|---|
| 1 Local | `docker compose up` (Postgres, Redis, MinIO, API, worker) + simulators/emulators |
| 2 Managed infra | Neon (Postgres), Upstash (Redis), Cloudflare R2 (storage). API and worker still local |
| 3 Host API | Railway / Render / Fly.io. Apps call real HTTPS |
| 4 Host worker | Railway / Fly.io (CPU), or Modal for GPU |
| 5 AWS (optional) | ECS Fargate, RDS, ElastiCache, S3, GPU worker |
