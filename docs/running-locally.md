# Running the Project

## Local Development (daily use)

Start only the database in Docker, then run the API and worker directly on your machine.
The API auto-restarts on file save via Spring DevTools.

```powershell
# 1. Start only the DB (once — leave it running)
docker compose up db -d

# 2. Run the API (auto-restarts on file save)
cd apps/api
.\mvnw.cmd spring-boot:run

# 3. Run the Python worker (separate terminal)
cd apps/worker
python services/chord_beat_analysis.py
```

To stop: `Ctrl+C` in each terminal. The DB keeps running until you explicitly stop it:

```powershell
docker compose down
# or just stop the DB container:
docker stop musicdecoded-db
```

---

## Deployment (full stack)

Builds and runs all services (DB, API, worker) as Docker containers.

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
| Docker      | Run containers          | DB (local dev) + full deployment |
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

> Redis and MinIO are added to `docker-compose.yml` by T-006. The full e2e script arrives with T-114.

## Services

| Service    | Local URL             |
| ---------- | --------------------- |
| API        | http://localhost:8080 |
| PostgreSQL | localhost:5432        |
