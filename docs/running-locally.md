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

# 3. Run the Python worker (separate terminal)
cd apps/worker
python services/chord_beat_analysis.py
```

When running outside Docker, use `localhost` instead of the service names in the env vars below
(e.g. `REDIS_URL=redis://localhost:6379`, `S3_ENDPOINT=http://localhost:9000`).

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
| Python 3.9+ | Run the worker          | Local dev                        |
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

> The full e2e script arrives with T-114.

## Services

| Service       | Local URL                                     | Dev credentials             |
| ------------- | --------------------------------------------- | --------------------------- |
| API           | http://localhost:8080                         | —                           |
| PostgreSQL    | localhost:5432 (db `musicdecoded`)            | `postgres` / `postgres`     |
| Redis         | redis://localhost:6379                        | none                        |
| MinIO S3 API  | http://localhost:9000 (bucket `musicdecoded`) | `minioadmin` / `minioadmin` |
| MinIO console | http://localhost:9001                         | `minioadmin` / `minioadmin` |

Postgres, Redis and MinIO listen on `127.0.0.1` only, because their dev credentials are public. The Android emulator (`10.0.2.2`) and iOS simulator still reach them; a physical phone on your Wi-Fi can reach the API (the Docker `api` service listens on all interfaces; a host-run API needs `--host 0.0.0.0` and a firewall rule for port 8080) but not MinIO, so file uploads need an emulator/simulator for now.

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
