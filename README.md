# MusicDecoded

Submit a YouTube link and get deep music analysis — beats, downbeats, chords, and a synced beat grid — replayed against the original YouTube video.

---

## What It Does

1. User submits a YouTube link
2. The system downloads temporary audio, runs ML models, and stores structured analysis artifacts
3. The Android app replays the YouTube video and syncs chord, beat, and waveform views to the player timestamp

The app does **not** permanently store downloaded audio or video — only metadata and analysis results.

---

## Stack

| Layer          | Technology                                       |
| -------------- | ------------------------------------------------ |
| Mobile         | Kotlin, Jetpack Compose                          |
| Backend        | Java, Spring Boot, Spring Security, JWT          |
| Database       | PostgreSQL                                       |
| Cache          | Redis                                            |
| Object Storage | MinIO (local) / Cloudflare R2 or S3 (production) |
| Queue          | Redis queue / Celery (local) → SQS (production)  |
| Worker         | Python, FFmpeg, PyTorch                          |
| ML Models      | Beat Transformer, CNN-LSTM chord recognition     |

---

## Project Structure

```
musicdecoded/
  apps/
    api/      # Java Spring Boot backend
    mobile/   # Kotlin Android app
    worker/   # Python ML worker
  infra/
    docker-compose.yml
  docs/
    architecture.md
    api-contracts.md
    data-model.md
    user-flows.md
    running-locally.md
```

---

## Running Locally

See [docs/running-locally.md](docs/running-locally.md) for the full setup guide.

**Quick start:**

```bash
# Start the database
docker compose up db -d

# Run the API
cd apps/api && ./mvnw.cmd spring-boot:run

# Run the worker
cd apps/worker && python services/chord_beat_analysis.py
```

---

## Docs

| Doc                                           | Description                                        |
| --------------------------------------------- | -------------------------------------------------- |
| [architecture.md](docs/architecture.md)       | System design, hosting plan, security rules        |
| [data-model.md](docs/data-model.md)           | Database schema, object storage layout, Redis keys |
| [api-contracts.md](docs/api-contracts.md)     | All API endpoints with request/response examples   |
| [user-flows.md](docs/user-flows.md)           | Core user flow, playback sync, Android screens     |
| [running-locally.md](docs/running-locally.md) | How to run the project locally vs full deployment  |
