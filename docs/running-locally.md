# Running the Project

## Local Development (daily use)

Start the backing services (Postgres, Redis, MinIO) in Docker, then run the API and worker directly on your machine.
The API auto-reloads on file save (`--reload`).

```powershell
# 1. Start the backing services (once — leave them running).
#    minio-init creates the `musicdecoded` bucket and exits.
docker compose up db redis minio minio-init -d

# 2. Run the API (auto-reloads on file save). Needs uv: https://docs.astral.sh/uv/
#    APP_ENV=dev supplies local defaults for JWT_SECRET and DATABASE_URL;
#    without it you must set both env vars or the API refuses to start.
cd apps/api
$env:APP_ENV = "dev"; uv run uvicorn app.main:app --reload --port 8080
#   macOS/Linux: APP_ENV=dev uv run uvicorn app.main:app --reload --port 8080
#   Tests: uv run pytest
#   Changed POSTGRES_USER/POSTGRES_PASSWORD in .env? The host-run API doesn't read .env, so also set
#   $env:DATABASE_URL = "postgresql://<user>:<password>@localhost:5432/musicdecoded"

# 3. Run the worker's queue consumer (separate terminal). Python 3.14 venv, CPU torch:
cd apps/worker
uv venv --python 3.14 .venv
uv pip install --python .venv -r requirements.txt -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
$env:REDIS_URL = "redis://127.0.0.1:6379"
$env:DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/musicdecoded"
$env:S3_ENDPOINT = "http://127.0.0.1:9000"; $env:S3_BUCKET = "musicdecoded"
$env:S3_ACCESS_KEY = "minioadmin"; $env:S3_SECRET_KEY = "minioadmin"
.venv\Scripts\python services/consumer.py
#   One-off analysis without the queue: python services/chord_beat_analysis.py data/sunnydays.mp3 outputs/out.json
```

When running outside Docker, use `127.0.0.1` instead of the service names in the env vars below
(e.g. `REDIS_URL=redis://127.0.0.1:6379`, `S3_ENDPOINT=http://127.0.0.1:9000`). `localhost` can resolve to
IPv6 first on Windows, and the containers only listen on IPv4.

The consumer normalizes audio with FFmpeg when it is on `PATH` (the Docker image has it) and otherwise falls
back to librosa decoding; it logs which one it used. It needs the chord model submodule
(`git submodule update --init`), or point `CHORD_MODEL_DIR` at a checkout of it. YouTube/SoundCloud jobs
need FFmpeg on `PATH` (librosa can't decode the `.webm`/`.m4a` that yt-dlp downloads); `requirements.txt`
installs yt-dlp with deno, the JS runtime YouTube extraction needs.

## End-to-end check

`scripts/e2e.sh` (bash: Git Bash on Windows, or CI) drives the whole pipeline through the API:
guest auth → `POST /uploads` → PUT `apps/worker/data/sunnydays.mp3` → `POST /jobs` → poll `GET /jobs/{id}`
(5 min timeout) → `GET /tracks/{id}/analysis` → download → `python contracts/validate.py`. It prints each step
and exits non-zero on the first failure.

```bash
# With the API and consumer running (above, or `docker compose up --build -d`):
PYTHON=apps/worker/.venv/Scripts/python scripts/e2e.sh   # any Python with jsonschema works
# Env: API_URL (default http://127.0.0.1:8080), AUDIO (another mp3), E2E_TIMEOUT (seconds, default 300)
```

The same mp3 is deduplicated by its SHA-256, so a second run is an instant cache hit (`200`, already
`complete`). Run `docker compose down -v` to start from scratch.

To stop: `Ctrl+C` in each terminal. The containers keep running until you explicitly stop them:

```powershell
docker compose down        # keeps data volumes
docker compose down -v     # also wipes Postgres and MinIO data
```

---

## Deployment (full stack)

Builds and runs all services (DB, Redis, MinIO, API, worker) as Docker containers.

```powershell
# From the project root
docker compose up --build -d

# Check everything is running
docker compose ps

# Watch logs
docker compose logs -f
```

To stop everything:

```powershell
docker compose down
```

> Only include `--build` when code or dependencies have changed since the last build.
> If you're just restarting containers with no changes, `docker compose up -d` is enough.

---

## Prerequisites

| Tool        | Purpose                 | Required for                     |
| ----------- | ----------------------- | -------------------------------- |
| Docker      | Run containers          | DB/Redis/MinIO (local dev) + full deployment |
| uv (Python 3.13) | Run and test the API | Local dev                  |
| Python 3.14 | Run the worker          | Local dev (same as the image)    |
| Android Studio | Build/run `apps/android` | Android lane                 |
| Xcode 16+ (macOS) | Build/run `apps/ios` | iOS lane                       |

## Mobile apps

- **Android**: open `apps/android` in Android Studio and run on an emulator. The emulator reaches the API at `http://10.0.2.2:8080`.
- **iOS** (macOS only): generate the project first: `brew install xcodegen && cd apps/ios && xcodegen generate`, then open `MusicDecoded.xcodeproj` and run on a simulator (see `apps/ios/README.md`). The simulator reaches the API at `http://localhost:8080`.

## Contracts

```powershell
pip install jsonschema
python contracts/validate.py                       # validates contracts/fixtures/*.json
npx @redocly/cli@1 lint contracts/openapi.yaml
```

## Services

| Service       | Local URL                                     | Dev credentials             |
| ------------- | --------------------------------------------- | --------------------------- |
| API           | http://localhost:8080                         | —                           |
| PostgreSQL    | localhost:5432 (db `musicdecoded`)            | `postgres` / `postgres`     |
| Redis         | redis://localhost:6379                        | none                        |
| MinIO S3 API  | http://localhost:9000 (bucket `musicdecoded`) | `minioadmin` / `minioadmin` |
| MinIO console | http://localhost:9001                         | `minioadmin` / `minioadmin` |

The API, Postgres, Redis and MinIO all listen on `127.0.0.1` only, because their dev secrets are public. The Android emulator (`10.0.2.2`) and iOS simulator still reach them; a physical phone cannot.

These are dev-only defaults from `docker-compose.yml`. To change them, copy `.env.example` to `.env`
(gitignored) and edit it; compose works with no `.env` at all.

> MinIO's official images are no longer published, so compose uses Bitnami's pinned archive builds
> (`bitnamilegacy/minio`, `bitnamilegacy/minio-client`).

### Android emulator and presigned URLs

Upload/download links point at `http://localhost:9000`. Inside the Android emulator `localhost` is the emulator itself, so forward the ports once per emulator boot:

```powershell
adb reverse tcp:9000 tcp:9000
adb reverse tcp:8080 tcp:8080   # optional: lets the app use localhost:8080 too
```

The iOS simulator shares the Mac's network, so `localhost` works there without this.

## Environment variables

Set by `docker-compose.yml` for the containers. Set them yourself when running the API or worker outside Docker.

| Variable                 | Used by     | Description |
| ------------------------ | ----------- | ----------- |
| `APP_ENV`                | api         | `dev` locally. Anything else (default `prod`) requires `JWT_SECRET` and `DATABASE_URL`. |
| `JWT_SECRET`             | api         | JWT signing key. Defaulted when `APP_ENV=dev`; required everywhere else. At least 32 bytes. Never commit it. |
| `DATABASE_URL`           | api, worker | Postgres URL, e.g. `postgresql://postgres:postgres@db:5432/musicdecoded`. Defaulted (localhost) for the api only when `APP_ENV=dev`. |
| `REDIS_URL`              | api, worker | Redis URL (job queue, job progress, caches), e.g. `redis://redis:6379`. |
| `S3_ENDPOINT`            | api, worker | S3-compatible endpoint. MinIO locally (`http://minio:9000`), R2 in prod. |
| `S3_BUCKET`              | api, worker | Bucket name, `musicdecoded`. |
| `S3_ACCESS_KEY`          | api, worker | S3 access key (MinIO root user locally). |
| `S3_SECRET_KEY`          | api, worker | S3 secret key (MinIO root password locally). |
| `OFFLINE_STREAM_AUDIO`   | api, worker | `false` by default. When `true`, streaming-source audio is kept for offline play (PLAN.md R-1). |
| `SOUNDCLOUD_ENABLED`     | api         | `false` by default. Enables the SoundCloud source. |
| `YOUTUBE_API_KEY`        | api         | YouTube Data API key for search. Set it in your own environment; never commit it. |
| `REVENUECAT_WEBHOOK_SECRET` | api      | Shared secret RevenueCat sends as `Authorization: Bearer <secret>` on `POST /billing/revenuecat`. Unset = the webhook returns 404 (billing off). Never commit it. |
