# Running the Project

## Local Development (daily use)

Start the backing services (Postgres, Redis, MinIO) in Docker, then run the API and worker directly on your machine.
The API auto-restarts on file save via Spring DevTools.

```powershell
# 1. Start the backing services (once — leave them running).
#    minio-init creates the `musicdecoded` bucket and exits.
docker compose up db redis minio minio-init -d

# 2. Run the API with the dev profile (auto-restarts on file save).
#    The dev profile supplies local defaults for JWT_SECRET and SPRING_DATASOURCE_PASSWORD;
#    without it you must set both env vars or the API refuses to start.
cd apps/api
.\mvnw.cmd spring-boot:run "-Dspring-boot.run.profiles=dev"
#   macOS/Linux: ./mvnw spring-boot:run -Dspring-boot.run.profiles=dev

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
| Java 21+    | Compile and run the API | Local dev                        |
| Python 3.9+ | Run the worker          | Local dev                        |
| Android Studio | Build/run `apps/android` | Android lane                 |
| Xcode 16+ (macOS) | Build/run `apps/ios` | iOS lane                       |

## Mobile apps

- **Android**: open `apps/android` in Android Studio and run on an emulator. The emulator reaches the API at `http://10.0.2.2:8080`.
- **iOS**: open `apps/ios/MusicDecoded.xcodeproj` (created by T-003) and run on a simulator. The simulator reaches the API at `http://localhost:8080`.

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

These are dev-only defaults from `docker-compose.yml`. To change them, copy `.env.example` to `.env`
(gitignored) and edit it; compose works with no `.env` at all.

> MinIO's official images are no longer published, so compose uses Bitnami's pinned archive builds
> (`bitnamilegacy/minio`, `bitnamilegacy/minio-client`).

## Environment variables

Set by `docker-compose.yml` for the containers. Set them yourself when running the API or worker outside Docker.

| Variable                 | Used by     | Description |
| ------------------------ | ----------- | ----------- |
| `SPRING_PROFILES_ACTIVE` | api         | `dev` locally. Outside `dev`, the API requires `JWT_SECRET` and `SPRING_DATASOURCE_PASSWORD`. |
| `JWT_SECRET`             | api         | JWT signing key. Defaulted by the `dev` profile; required everywhere else. Never commit it. |
| `SPRING_DATASOURCE_URL` / `_USERNAME` / `_PASSWORD` | api | Postgres JDBC connection. Password defaulted only by the `dev` profile. |
| `DATABASE_URL`           | worker      | Postgres URL, e.g. `postgresql://postgres:postgres@db:5432/musicdecoded`. |
| `REDIS_URL`              | api, worker | Redis URL (job queue, job progress, caches), e.g. `redis://redis:6379`. |
| `S3_ENDPOINT`            | api, worker | S3-compatible endpoint. MinIO locally (`http://minio:9000`), R2 in prod. |
| `S3_BUCKET`              | api, worker | Bucket name, `musicdecoded`. |
| `S3_ACCESS_KEY`          | api, worker | S3 access key (MinIO root user locally). |
| `S3_SECRET_KEY`          | api, worker | S3 secret key (MinIO root password locally). |
| `OFFLINE_STREAM_AUDIO`   | api, worker | `false` by default. When `true`, streaming-source audio is kept for offline play (PLAN.md R-1). |
| `SOUNDCLOUD_ENABLED`     | api         | `false` by default. Enables the SoundCloud source. |
| `YOUTUBE_API_KEY`        | api         | YouTube Data API key for search. Set it in your own environment; never commit it. |
