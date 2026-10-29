# MusicDecoded — Product & Architecture Plan

> Read this file before any task. It defines **what** we're building and **why**. `TASKS.md` defines the work, `AGENTS.md` defines how agents work, and `contracts/` defines the interfaces.

## 1. Vision

MusicDecoded turns any song into chords you can follow. A user picks a song from their files, records it, or finds it on YouTube or SoundCloud. They get a scrolling player that shows the spectrogram, the chord changes, the key, the BPM and the time signature, plus a piano that plays along. Everything they transcribe is saved on their phone and works offline. An account is optional and only adds sync across devices.

## 2. Platforms

| App | Stack | Location |
|---|---|---|
| iOS | Swift 6, SwiftUI, SwiftData, AVFoundation, URLSession async/await (iOS 17+) | `apps/ios` |
| Android | Kotlin, Jetpack Compose, Room, Media3 ExoPlayer, Ktor client (minSdk 26) | `apps/android` |
| API | Python 3.13, FastAPI, SQLAlchemy 2, PyJWT, PostgreSQL (uv for dependencies) | `apps/api` |
| Worker | Python, FFmpeg, PyTorch (Beat Transformer, CNN-LSTM chords), librosa | `apps/worker` |

**Two separate native apps, no shared code.** Consistency between them comes from `contracts/` (API, analysis file format, chord voicings, fixtures) and from `DESIGN.md`, not from a shared codebase.

## 3. V1 scope

### 3.1 Home, Library, navigation
Visuals for everything in §3 are in `DESIGN.md`, translated from the Figma prototype.
- Tabs: **Home · Library · Settings** (a 4th **Tuner** tab is in the prototype but awaits a decision: T-505). A **mini player** sits above the tabs.
- **Home**: "Recently Decoded" (the 3 newest history rows) plus **Transcribe Music**: Files, Media Library, YouTube, SoundCloud, and a Record card. Rows show the status (`analyzing`, `ready`, `failed`), an offline badge, and `duration · key · BPM`.
- **Library**: the full history, with search (title/artist/key), filters (All · Favorites · Offline · Recordings) and counts. Items can be **favorited**.
- **Processing** screen: progress mapped from `Job.stage`, with "Continue in background".

### 3.2 Sources (`MediaSource`)

| Source | How the user picks it | What gets analyzed | Offline audio on device |
|---|---|---|---|
| Files | System file picker / media library (`UIDocumentPicker`/`MPMediaPicker`, Android `ACTION_OPEN_DOCUMENT`) | Uploaded copy of the file | **Yes**: the file is copied into app storage |
| Record | In-app recorder (AVAudioRecorder / MediaRecorder, AAC 44.1k mono) | Uploaded recording | **Yes** |
| YouTube | In-app "sandbox": search screen → results → tap a video | Worker fetches the audio server-side by `videoId` | **Flag-gated** (`offlineStreamAudio`, see R-1). Default: the chord file is offline; playback streams through the official player |
| SoundCloud | Paste a link (v1); search later | Worker fetches by URL | Flag-gated, same as YouTube |

Every source implements the same flow: **pick → (upload) → `POST /jobs` → poll → download analysis → save locally → open player.**

### 3.3 Player (v1; details and visuals in `DESIGN.md`)
- **Top**: key, time signature and BPM chips, with room for more.
- **Timeline**: scrolls left to right under a fixed playhead. A chord track (tap a block to seek) sits over the waveform (`waveformPeaks`; spectrogram is an optional alternate view). Drag to scrub, pinch to zoom.
- **Now playing**: a big current-chord name plus its note names.
- **Control chips**: Chords (Basic/Full simplification), Voicing, Transpose (display only in v1).
- **Bottom panel**, toggling **Piano** (keyboard highlighting the chord from `contracts/chord-voicings.json`) and **Video** (YouTube embed synced to the timeline; youtube sources only). An expand button opens the **Piano learning view** (large keyboard, up-next chord, progression strip).
- **Transport**: speed (time-stretch, pitch kept), previous/next chord, play/pause, loop (current bar or a dragged range).

### 3.4 Accounts (optional)
- **Guest (default)**: on first launch the app calls `POST /auth/anonymous` with a random `deviceId`. The library lives only on the device and is not transferable.
- **Account**: signup/login. Logging in on a device that has guest data **links** that data to the account. With an account, the library and analysis files sync across devices, and so does audio the user owns (files and recordings). Streaming-source audio is never synced; it is re-fetched if the flag allows.

## 4. Non-goals for v1
On-device ML inference, lyrics, sheet music export, a beat-grid view, a light theme (v1 is dark-only), transposing the audio pitch, a tuner (pending T-505), social or sharing features, editing chords, payments/Pro, a web client, iPad/tablet-specific layouts, and SoundCloud search.

## 5. Architecture

```
 iOS app (SwiftUI)          Android app (Compose)
   local: SwiftData +          local: Room +
   files/audio, analysis/      files/audio, analysis/
          \                        /
           \   HTTPS / REST (contracts/openapi.yaml)
            v                    v
        ┌──────────────── FastAPI API ──────────────┐
        │ auth (guest+account JWT) · uploads · jobs │
        │ youtube search proxy · sync · config      │
        └──┬──────────┬──────────────┬──────────────┘
           │          │              │
      PostgreSQL   Redis queue   Object storage (MinIO local / R2 prod)
                      │            uploads/  analysis/  extracted-audio/
                      v              ^
                Python worker ───────┘
                FFmpeg → beats → chords → key → bpm/timesig → spectrogram
                → analysis.json (contracts/analysis.schema.json)
```

Rules:
1. Apps call **only** the API. Object storage is reached only through presigned URLs the API hands out.
2. **The device is the source of truth for a user's library.** The server keeps analysis files (deduped globally by content hash, videoId or URL) and, for accounts only, a synced copy of library metadata.
3. Analysis is **global and cached**. Two users who transcribe the same YouTube video or byte-identical file share one analysis.
4. The worker never talks to apps. It reads jobs from Redis, writes results to object storage, and writes status to Postgres and Redis.

### 5.1 On-device storage (both apps, same layout)
```
<app support dir>/
  audio/{trackId}.{ext}        # files, recordings, flag-gated stream audio
  analysis/{trackId}.json      # analysis v1 file, verbatim from the server
  db                           # Room / SwiftData: LibraryItem rows (mirrors contracts LibraryItem)
```
The player must work with **no network** whenever `analysis/{trackId}.json` exists. If `audio/` is missing (streaming source with the flag off), the player shows the timeline and piano and needs network for audio/video.

### 5.2 Sync
- Account tokens only. `GET /sync/library?since=cursor` and `PUT /sync/library`.
- Conflicts resolve **last-write-wins on `updatedAt`**. Deletes are tombstones (`deleted: true`).
- When a pulled item has no local analysis, the app downloads it via `GET /tracks/{id}/analysis`. When a pulled file/recording has no local audio, the app downloads it using `audioUploadId`.
- Triggers: app foreground, after a job completes, and after login.

### 5.3 Analysis pipeline (worker)
Normalize to mono 22.05 kHz WAV → beats + downbeats (librosa on the free tier; Beat Transformer on Pro, see `apps/worker/MODELS.md`) → chords (CNN-LSTM) → key (Krumhansl-Schmuckler on mean chroma) → BPM (60 / median inter-beat interval) → time signature (median beats per downbeat interval, denominator 4) → mel spectrogram (64 bins, 20 fps, uint8) and waveform peaks (20 fps) → write `analysis.json` → delete temp media.

## 6. Decisions

| ID | Decision | Why |
|---|---|---|
| D-1 | Analysis runs server-side, reusing the existing worker | The models already work in Python. Converting them for on-device use (CoreML/TFLite) is a v2 project. The analysis file is the seam where on-device inference could slot in later. |
| D-2 | Guests use anonymous device JWTs | Analysis needs the server, rate limiting needs an identity, and there's no friction for the user. |
| D-3 | All sources go through one `MediaSource` → job → analysis flow | One pipeline to test. New sources are cheap to add. |
| D-4 | YouTube "sandbox" = server-proxied Data API search + official embedded player (YouTube IFrame Player API in a WKWebView / android-youtube-player) | The API key stays secret, and we stay within YouTube's embed terms. |
| D-5 | Contracts-first: `contracts/` changes land before code that uses them | Two native apps plus a backend built by parallel agents need a fixed interface. |
| D-6 | No shared mobile code (no KMP) | The user chose this. Each app is idiomatic to its platform. |
| D-7 | Piano and timeline are drawn with native Canvas (SwiftUI `Canvas`, Compose `Canvas`), with no third-party libraries | Both are simple 2D drawing, and it avoids dependency risk. |
| D-8 | The worker fetches YouTube/SoundCloud audio with **yt-dlp** as a Python library (audio-only, `bestaudio`), pinned and updated regularly | The most robust, best-maintained extractor (Unlicense), and it handles both sources. Rejected: `zararashraf/youtube-video-downloader-api`, a separate Flask service wrapping `pytubefix` that downloads progressive *video*: an extra hop, more bytes, a weaker extractor. |
| D-9 | Licensing: apps, contracts and docs are **MIT** (root `LICENSE`); `apps/api` and `apps/worker` are **AGPL-3.0** | MIT keeps the apps easy to read and contribute to (recruiters, contributors). AGPL stops someone from hosting a closed clone of the backend. The owner holds copyright, so the official hosted service and Pro are unaffected. Contributions to AGPL parts need a CLA if relicensing is ever wanted (T-603). |
| D-11 | The API is **FastAPI (Python)**, rewritten from Spring Boot before M1 | Hosting cost: ~70 MB RAM per container vs ~300–500 MB for Spring Boot, so it fits the cheapest tiers and scales to zero well. One backend language with the worker. Measured: 67 MB after requests. |
| D-10 | Pro is enforced **server-side** via the user's `plan`; the apps only reflect it (see §9) | The apps are open source, so client checks can be patched out. What costs money (analysis, models, storage) only exists on the server. |

## 7. Risks

| ID | Risk | Mitigation |
|---|---|---|
| **R-1** | **Saving YouTube/SoundCloud audio on the device breaks YouTube's ToS and SoundCloud's terms. App Store and Play Store reviewers reject apps that do this.** | Offline audio for streaming sources sits behind the server flag `offlineStreamAudio`, **false for store builds**. The chord file is always offline. Playback uses the official players. The product owner decides before store submission (T-501). |
| R-2 | Model latency and cost (GPU) | Global analysis cache, per-user rate limits, a 10-minute max duration in v1, and a CPU worker. **Measure before pricing: T-115** produces seconds-per-song and cost per 1,000 songs. |
| R-3 | SoundCloud API access is restricted for new apps | SoundCloud is milestone M5 and behind `soundcloudEnabled`. Fallback: paste a link, and the worker resolves it. |
| R-4 | Two apps drift apart in behavior or design | Shared fixtures, `DESIGN.md`, and parity tasks: every app task has an iOS twin and an Android twin. |
| R-5 | The secret JWT fallback in `application.properties` | Removed in T-004. The app fails to start without `JWT_SECRET` outside the `dev` profile. |
| **R-6** | **YouTube Data API quota.** The default is 10,000 units/day and `search.list` costs 100, so **~100 searches/day for the whole app**. This hits long before compute cost does | Redis cache per query (T-301); suggestions from local recent queries; "Paste a URL" needs no search. **Apply for a quota increase early (T-306)**, since it needs a working app + privacy policy and takes weeks. |
| **R-7** | **madmom's pretrained models are CC BY-NC-SA 4.0 (non-commercial).** The worker used madmom as a beat-tracking fallback. Using it in a paid product would violate the license | **Done (T-116):** madmom removed, and a test fails if it comes back (librosa is the primary path; Beat Transformer is MIT). Its source code (BSD) is fine, but the bundled models aren't. Check every new model's **weights** license, not just its code license. |
| R-8 | yt-dlp gets blocked from datacenter IPs ("Sign in to confirm you're not a bot") | Pin and auto-update yt-dlp; retry with backoff; surface `failed` with a clear message. If it becomes chronic: residential egress or PO-token setup, decided at that time. Note this doesn't change R-1. |

## 8. Milestones

| M | Goal | Exit criteria |
|---|---|---|
| M0 Foundation | Repo layout, contracts, CI, iOS project exists | Both apps build in CI; fixture validates |
| M1 Pipeline | File upload → worker → analysis v1 → download, end to end with curl | `docs/running-locally.md` e2e script passes |
| M2 Apps core | Both apps: guest auth, history, Files + Record sources, player against the fixture and real jobs | Transcribe a local mp3 on both apps, then play it in airplane mode |
| M3 YouTube | Sandbox search, embedded video, worker fetch | Search → tap → chords → video synced on both apps |
| M4 Accounts & sync | Signup/login/link, sync | Transcribe on device A, see it on device B |
| M5 Polish & store | SoundCloud, empty/error states, accessibility, store checklist | TestFlight + Play internal track builds |
| M6 Pro | Entitlements, purchases, first Pro features | A sandbox purchase on both stores flips `plan` to `pro` and raises limits |

## 9. Open source & monetization

**The repo stays public. Pro is a server-side entitlement, not a separate codebase.** A private module gets added only if a Pro feature ever needs secret client-side logic (none planned).

### Plans
| | Free | Pro |
|---|---|---|
| Transcriptions | Daily limit (guest 10, account 50; tune after T-115) | Higher or unlimited |
| Analysis models | Standard pipeline (librosa beats + CNN-LSTM chords) | Premium: Beat Transformer, stem separation, large-vocabulary chords (GPU) |
| Sync | Library + chord files | + sync of the user's own audio (storage cost) |
| Learning | Player, piano, speed, loop, transpose | Extras: practice stats, chord sheet / MIDI export (post-v1) |

Rule: **a feature is Pro if it costs us money per use** (compute, storage) or is a clear power-user extra. Core playback and learning stay free.

### Mechanics
- `users.plan` (`free` | `pro`) on the server, also exposed as `ClientConfig.plan` + `limits` and as the JWT `plan` claim. The **API enforces** limits and model choice; the apps only show or hide UI and upsell.
- **Purchases must use the App Store / Google Play billing** for digital features (15% under the small-business programs). Use **RevenueCat** to unify both stores: the app purchases through the RevenueCat SDK, and RevenueCat's webhook calls the API (`POST /billing/revenuecat`, added in M6), which sets `plan`.
- Guests can't buy Pro. Purchasing requires an account so the entitlement survives reinstalls and spans devices.

### Cost model (fill in from T-115)
`monthly compute ≈ (new, uncached songs/day × seconds/song ÷ 3600) × $/worker-hour × 30`. Cache hits cost ~0. Storage (R2) ≈ $0.015/GB-month, at ≤1.5 MB per analysis file. Re-evaluate Pro limits once real numbers exist.
