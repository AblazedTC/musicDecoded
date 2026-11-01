# MusicDecoded

[![License: MIT / AGPL-3.0](https://img.shields.io/badge/license-MIT%20%2F%20AGPL--3.0-blue)](#license)
[![contracts](https://github.com/AblazedTC/musicDecoded/actions/workflows/contracts.yml/badge.svg)](https://github.com/AblazedTC/musicDecoded/actions/workflows/contracts.yml)
[![api](https://github.com/AblazedTC/musicDecoded/actions/workflows/api.yml/badge.svg)](https://github.com/AblazedTC/musicDecoded/actions/workflows/api.yml)
[![worker](https://github.com/AblazedTC/musicDecoded/actions/workflows/worker.yml/badge.svg)](https://github.com/AblazedTC/musicDecoded/actions/workflows/worker.yml)
[![android](https://github.com/AblazedTC/musicDecoded/actions/workflows/android.yml/badge.svg)](https://github.com/AblazedTC/musicDecoded/actions/workflows/android.yml)
[![ios](https://github.com/AblazedTC/musicDecoded/actions/workflows/ios.yml/badge.svg)](https://github.com/AblazedTC/musicDecoded/actions/workflows/ios.yml)

Turn any song into chords you can follow. Pick a file, record audio, or find a song on YouTube or SoundCloud. MusicDecoded transcribes the beats, chords, key, BPM and time signature, then shows them on a scrolling player with a piano that plays along. Transcriptions are saved on your phone and work offline. An account is optional and only adds sync across devices.

<p align="center">
  <img src="docs/screenshots/home.png" width="200" alt="Home: recent songs and sources">
  <img src="docs/screenshots/player.png" width="200" alt="Player: scrolling chord timeline over the waveform, piano with fingering">
  <img src="docs/screenshots/chord-sheet.png" width="200" alt="Chord sheet view with Roman numerals">
  <img src="docs/screenshots/piano-focus.png" width="200" alt="Piano focus: current and next chord, inversions">
</p>

**Features**
- Native **iOS (SwiftUI)** and **Android (Jetpack Compose)** apps that share one API contract
- Sources: files, the music library, the microphone, an in-app YouTube browser, SoundCloud links
- Four player views: timeline, beat grid, chord sheet and piano focus, with Roman numerals, transpose,
  simplified chords, and piano voicings with inversions and fingering
- Local-first: audio and chord files are stored on the device and play offline; accounts only add sync
- Python worker: beat tracking, a CNN-LSTM chord model, key, BPM and time signature detection
- Runs on one small server with Docker Compose ([deploy/README.md](deploy/README.md))

## Architecture

```mermaid
flowchart TD
    ios["iOS app (SwiftUI)<br/>SwiftData + local audio/analysis"]
    android["Android app (Compose)<br/>Room + local audio/analysis"]
    api["FastAPI API<br/>auth (guest + account JWT) · uploads · jobs<br/>sync · config"]
    pg[(PostgreSQL)]
    redis[(Redis queue)]
    s3[("Object storage<br/>MinIO or any S3")]
    worker["Python worker<br/>FFmpeg → beats → chords → key → BPM/time sig → spectrogram<br/>→ analysis.json"]

    ios -- "HTTPS / REST (contracts/openapi.yaml)" --> api
    android -- "HTTPS / REST" --> api
    api --> pg
    api --> redis
    api -- presigned URLs --> s3
    redis --> worker
    worker --> s3
    worker --> pg
```

Apps talk only to the API; audio and analysis files move through presigned object-storage URLs. Analysis is
cached globally, so a song that was already transcribed costs nothing. Details in [PLAN.md §5](PLAN.md#5-architecture).

Want to help? Start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Start here

| File | What it's for |
|---|---|
| [PLAN.md](PLAN.md) | Vision, v1 scope, architecture, decisions, risks, milestones |
| [TASKS.md](TASKS.md) | The work board |
| [DESIGN.md](DESIGN.md) | Screens and player spec |
| [contracts/](contracts) | API (`openapi.yaml`), analysis file format, chord voicings, fixtures |

## Stack

| Layer | Technology |
|---|---|
| iOS | Swift, SwiftUI, SwiftData, AVFoundation |
| Android | Kotlin, Jetpack Compose, Room, Media3 |
| API | Python, FastAPI, SQLAlchemy, JWT |
| Worker | Python, FFmpeg, PyTorch (Beat Transformer, CNN-LSTM chords) |
| Data | PostgreSQL, Redis, S3-compatible storage (MinIO / R2) |

## Structure

```
apps/
  android/   Kotlin Android app
  ios/       Swift iOS app
  api/       FastAPI API
  worker/    Python ML worker
contracts/   shared interfaces: openapi.yaml, analysis.schema.json, chord-voicings.json, fixtures/
docs/        architecture, data model, user flows, running locally
```

## Running locally

See [docs/running-locally.md](docs/running-locally.md).

```bash
git clone --recurse-submodules https://github.com/AblazedTC/musicDecoded.git && cd musicDecoded
docker compose up --build -d          # API on http://localhost:8080, worker, Postgres, Redis, MinIO
```

Then run the Android app from Android Studio (emulator) or the iOS app from Xcode (simulator); both point at
the local API in debug builds. The debug builds also include a sample song: long-press the Home title.

To run it on a server, see [deploy/README.md](deploy/README.md).

## License

| Part | License |
|---|---|
| iOS app, Android app, `contracts/`, docs | [MIT](LICENSE) |
| `apps/api`, `apps/worker` | [AGPL-3.0](apps/api/LICENSE) |

Contributions to `apps/api` and `apps/worker` need a signed CLA (see [CONTRIBUTING.md](CONTRIBUTING.md#licensing-and-the-cla)).
"MusicDecoded" and its logo are not covered by these licenses. Forks must use a different name. See [PLAN.md §9](PLAN.md#9-open-source--monetization) for how the free and Pro plans work.

## Docs

| Doc | Description |
|---|---|
| [architecture.md](docs/architecture.md) | Components, request lifecycle, security, hosting |
| [data-model.md](docs/data-model.md) | On-device storage, Postgres, object storage, Redis, queue message |
| [user-flows.md](docs/user-flows.md) | Step-by-step flows for every feature |
| [running-locally.md](docs/running-locally.md) | Local setup |
