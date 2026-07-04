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

## Services

| Service    | Local URL             |
| ---------- | --------------------- |
| API        | http://localhost:8080 |
| PostgreSQL | localhost:5432        |
