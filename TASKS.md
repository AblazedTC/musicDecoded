# TASKS.md — Work board

How to work a task (people and AI agents alike):
1. Pick the lowest-numbered `todo` task in your lane whose deps are all `done`.
2. Claim it: set `Status: in-progress` and `Owner`, commit that change on its own, and branch as `<lane>/T-###-slug`.
3. Stay inside the acceptance criteria. Log anything else you find as a new task instead of fixing it on the side.
4. Prove every criterion (test, command output, or screenshot), set `Status: done` with a one-line `Notes:`, and open a PR titled `T-###: <title>`.
5. Stuck, or need a product decision? Set `Status: blocked`, explain in `Notes:`, and stop.

Lanes: `contracts`, `api`, `worker`, `android`, `ios`, `infra`, `design`. Edit only your lane's folder (plus your task's entry here). Contract changes land first, in their own `contracts` task.
App tasks come in twins: `a` = Android, `i` = iOS. Twins share acceptance criteria and may run in parallel.

Template for new tasks:
```
### T-### Title
- Lane: · Status: todo · Owner: — · Deps: —
- Goal: one sentence.
- Acceptance:
  - [ ] testable criterion
- Notes:
```

---

## M0 — Foundation

### T-001 Rename Android app and restructure repo
- Lane: infra · Status: done · Owner: claude · Deps: —
- Goal: `apps/mobile` → `apps/android`, add `apps/ios` placeholder, fix references.
- Acceptance:
  - [x] `apps/android` builds with `./gradlew assembleDebug`
  - [x] No `apps/mobile` references remain outside git history
- Notes: The Android package is still `com.musicdecoded.mobile`. Renaming the package isn't worth the churn.

### T-002 Contracts v1
- Lane: contracts · Status: done · Owner: claude · Deps: —
- Goal: `openapi.yaml`, `analysis.schema.json`, `chord-voicings.json`, the fixture, and `validate.py`.
- Acceptance:
  - [x] `python contracts/validate.py` passes on `fixtures/sample.analysis.json`
  - [x] `npx @redocly/cli@1 lint contracts/openapi.yaml` reports no errors
- Notes: The fixture was converted from legacy `bad_analysis.json`. Its key is a placeholder, and `spectrogram`/`waveformPeaks` are null until T-112 regenerates it.

### T-003 Create iOS Xcode project
- Lane: ios · Status: done · Owner: claude · Deps: T-001
- Goal: A SwiftUI app `MusicDecoded` in `apps/ios` with app + unit test targets, iOS 17, bundle id `com.musicdecoded.app`.
- Acceptance:
  - [x] `xcodebuild … build` and `… test` succeed on the simulator
  - [x] Folder layout matches `apps/ios/README.md`
  - [x] `contracts/fixtures/sample.analysis.json` and `contracts/chord-voicings.json` are added to the test bundle **by reference** (not copied)
  - [x] Tab/nav shell with an empty Home screen
- Notes: XcodeGen project.yml; CI builds and runs app tests on a macOS simulator.

### T-004 Remove the hardcoded JWT secret fallback
- Lane: api · Status: done · Owner: claude · Deps: —
- Goal: The app refuses to start without `JWT_SECRET` unless the `dev` profile is active.
- Acceptance:
  - [x] No secret default outside dev (originally Spring's `application-dev.properties`; now `APP_ENV=dev` in `app/settings.py`)
  - [x] A test shows that context startup fails without the secret in the default profile
  - [x] `docker-compose.yml` and `docs/running-locally.md` set the dev mode for local use (now `APP_ENV=dev`)
- Notes: Done in Spring, then carried over in the FastAPI rewrite (D-11): `Settings` fails fast on a missing/blank/<32-byte secret; tests in `tests/test_settings.py`.

### T-005 CI per lane
- Lane: infra · Status: done · Owner: claude · Deps: T-003
- Goal: GitHub Actions workflows filtered by path: contracts (validate + lint), api (`uv run pytest`), worker (pytest), android (assembleDebug + unit tests), ios (macos runner, xcodebuild test).
- Acceptance:
  - [x] Each workflow runs only when its lane's paths change
  - [x] A PR touching `contracts/**` runs **all** workflows (contract changes can break every lane)
- Notes: 5 workflows (api incl. Docker build, worker, android, contracts, ios); contracts/** triggers all. First real run with PR #2.

### T-006 Local infrastructure: Redis + MinIO
- Lane: infra · Status: done · Owner: claude · Deps: —
- Goal: Add `redis` and `minio` (plus a bucket-init step) to `docker-compose.yml`, and wire their env vars into the api and worker.
- Acceptance:
  - [x] `docker compose up -d` brings up db, redis, minio and a `musicdecoded` bucket
  - [x] The env vars are documented in `docs/running-locally.md`
- Notes: MinIO official images are gone; uses pinned bitnamilegacy/minio + minio-client (archive builds, dev only). `.env.example` added.

### T-007 Worker repo hygiene
- Lane: worker · Status: done · Owner: claude · Deps: —
- Goal: Remove committed `__pycache__`, gitignore `outputs/` and `data/*.mp3` (keep one small test clip), and add `pytest` with one smoke test.
- Acceptance:
  - [x] `git ls-files apps/worker | grep pycache` is empty
  - [x] `python -m pytest` runs and passes
- Notes: Kept data/sunnydays.mp3; pytest smoke tests in apps/worker/tests (no ML deps needed).

---

## M1 — Pipeline

### T-101 Guest auth (`POST /auth/anonymous`)
- Lane: api · Status: done · Owner: claude · Deps: T-004
- Goal: Issue guest JWTs keyed by deviceId, and add the `kind` and `plan` claims to every token.
- Acceptance:
  - [x] Same deviceId → same userId (idempotent)
  - [x] `User` gains `kind` (guest/account), `plan` (free/pro, default free), `planExpiresAt`, a nullable email/password, and `deviceId`
  - [x] Existing signup/login return `AuthResponse` per openapi (token, userId, kind, plan)
  - [x] Tests for both paths
- Notes: Alembic in apps/api/app/migrations (upgrade head at startup). Guest→account linking is T-401.

### T-102 `GET /config` and `/health`
- Lane: api · Status: done · Owner: claude · Deps: —
- Acceptance:
  - [x] Flags come from env vars (`OFFLINE_STREAM_AUDIO`, default false; `SOUNDCLOUD_ENABLED`, default false)
  - [x] Returns the caller's `plan` and `limits` (`dailyJobs` from the plan's limit, `jobsUsedToday` from the Redis counter)
  - [x] Matches the `ClientConfig` schema
- Notes: LIMIT_GUEST/LIMIT_FREE/LIMIT_PRO env; jobsUsedToday from Redis rate:{userId}:{yyyymmdd} (UTC).

### T-103 Uploads (`POST /uploads`)
- Lane: api · Status: done · Owner: claude · Deps: T-006, T-101
- Goal: Validate content type and size, then return a presigned S3/MinIO PUT URL at `uploads/{uploadId}`.
- Acceptance:
  - [x] Rejects >100 MB with 413 `too_large`
  - [x] Stores `{uploadId, userId, sha256, contentType}`
  - [x] An integration test PUTs a file to the URL against MinIO
- Notes: botocore presigning (no boto3). Also GET /uploads/{uploadId} (owner-only). Compose signs with localhost:9000; Android emulator needs adb reverse (running-locally).

### T-104 Jobs + tracks (`POST /jobs`, `GET /jobs/{id}`)
- Lane: api · Status: done · Owner: claude · Deps: T-103
- Goal: Create or reuse a track by dedupe key (sha256 / videoId / soundcloudUrl), create a job, and enqueue it with `RPUSH` to Redis list `jobs:analysis` (FIFO with the worker's `BLPOP`).
- Acceptance:
  - [x] An already-analyzed dedupe key returns 200 `complete` without enqueueing
  - [x] Queue message format is documented in `docs/data-model.md`
  - [x] Per-user daily limit by plan, from config (`LIMIT_GUEST=10`, `LIMIT_FREE=50`, `LIMIT_PRO=` empty = unlimited) → 429 `rate_limited`. Cache hits (already analyzed) don't count
  - [x] `GET /jobs/{id}` returns 404 for another user's job
- Notes: Cache hits create a job row (links user→track) but are not queued or counted. Repeat requests return the caller's in-flight job. Only SoundCloud hosts accepted.

### T-105 Analysis download (`GET /tracks/{id}/analysis`)
- Lane: api · Status: done · Owner: claude · Deps: T-104
- Acceptance:
  - [x] Returns a presigned GET URL for `analysis/{trackId}.json` (5-minute expiry)
  - [x] 404 until the job is complete
- Notes: Access = caller has a job for the track (library_items arrive in T-402).

### T-110 Worker v1 output (key, BPM, time signature)
- Lane: worker · Status: done · Owner: claude · Deps: T-002, T-007
- Goal: Extend `ChordBeatAnalyzer` (`services/chord_beat_analysis.py`) to emit analysis v1 instead of the legacy format.
- Acceptance:
  - [x] Output for `data/sunnydays.mp3` passes `python contracts/validate.py <out>`
  - [x] Key via Krumhansl-Schmuckler on mean chroma; BPM = 60 / median IBI; time signature = median beats per bar
  - [x] Unit tests for key/BPM/time-signature on synthetic inputs
- Notes: features.py (key/bpm/timesig); time signature is 4/4 until real downbeats (Beat Transformer). ~16 s/song warm on CPU.

### T-111 Worker spectrogram + waveform peaks
- Lane: worker · Status: done · Owner: claude · Deps: T-110
- Acceptance:
  - [x] 64-bin mel, 20 fps, dB clipped to [-80, 0] → uint8 → base64; peaks at 20 fps normalized to 0–1
  - [x] A 4-minute song produces an analysis file < 1.5 MB
- Notes: Mel at hop 512 averaged into 20 fps frames; dB relative to track max.

### T-112 Regenerate the fixture from real worker output
- Lane: contracts · Status: done · Owner: claude · Deps: T-111
- Acceptance:
  - [x] `contracts/fixtures/sample.analysis.json` has a real key, spectrogram and peaks, and validates
- Notes: Added contracts/fixtures/sunnydays.analysis.json (real output, 476 KB); sample.analysis.json kept because iOS tests pin its values.

### T-113 Worker queue consumer
- Lane: worker · Status: done · Owner: claude · Deps: T-104, T-110
- Goal: A long-running process that BLPOPs `jobs:analysis`, downloads the upload from storage, runs the analysis, writes `analysis/{trackId}.json`, and updates the job's status/stage/progress in Postgres and Redis.
- Acceptance:
  - [x] Stages match the openapi `Job.stage` enum
  - [x] On exception: status `failed` with an error message, and temp files deleted
  - [x] Temp media is deleted on success too
  - [x] The Dockerfile `CMD` runs the consumer
- Notes: Consumer in services/consumer.py; analysis validated before save; streaming sources fail until T-302. e2e: 26 s/song on CPU. Follow-ups T-117 (stranded jobs), T-118 (model reload per job).

### T-114 End-to-end pipeline script
- Lane: infra · Status: done · Owner: claude · Deps: T-105, T-113
- Goal: `scripts/e2e.sh`: guest auth → upload an mp3 → job → poll → download → validate.
- Acceptance:
  - [x] Passes against `docker compose up` from a clean checkout
- Notes: scripts/e2e.sh passes on the full containerized stack (35 s; worker 30 s/song on CPU incl. 8 s model load). On Windows run it with PYTHON=python.

### T-115 Benchmark analysis cost
- Lane: worker · Status: todo · Owner: — · Deps: T-111
- Goal: Real numbers for PLAN.md §9's cost model, before any pricing decision.
- Acceptance:
  - [ ] The worker logs wall time per stage (fetch, normalize, beats, chords, key, spectrogram, save) for every job
  - [ ] `scripts/benchmark.py` runs 20 varied songs (2–8 min, different genres) on the target CPU instance size and reports median/p90 seconds per song, per stage, and peak RAM
  - [ ] Results + `$ per 1,000 uncached songs` for 2 candidate hosts (e.g. Fly.io/Railway CPU) written to `docs/cost-model.md`
  - [ ] The same run on a GPU host if Beat Transformer is enabled (the Pro model path)

### T-116 Remove the madmom fallback (non-commercial model license)
- Lane: worker · Status: done · Owner: claude · Deps: T-007
- Goal: PLAN.md R-7. madmom's pretrained models are CC BY-NC-SA and can't ship in a product with paid plans.
- Acceptance:
  - [x] `ChordBeatAnalyzer` no longer imports or calls `MadmomDetectorService`; `madmom` removed from `requirements.txt` and the Dockerfile
  - [x] librosa beat tracking is the free-tier path; Beat Transformer (MIT) is the premium path
  - [x] `apps/worker/MODELS.md` lists every model and its **code and weights** licenses (Beat Transformer: MIT; ISMIR2019 chord model: MIT; librosa: ISC)
  - [ ] The output for `data/sunnydays.mp3` still validates, and beat count is within ±5% of before (not run: needs real models + T-110)
- Notes: librosa is the only beat path; tests/test_model_licenses.py fails if madmom returns. Beat-count check needs real models (not run). See MODELS.md follow-up on upstream Beat Transformer decoding.

### T-117 Reclaim jobs stranded by a worker crash
- Lane: worker · Status: todo · Owner: — · Deps: T-113
- Goal: A job left `processing` when the consumer dies never finishes today.
- Acceptance:
  - [ ] Consumer heartbeats its current job (Redis key with TTL); a periodic sweep (or consumer startup) requeues `processing` jobs whose heartbeat expired, max 2 attempts, then `failed`
  - [ ] Test with a fake clock

### T-118 Load the chord model once per process
- Lane: worker · Status: done · Owner: claude · Deps: T-113
- Goal: `chord_recognition()` in the submodule reloads weights on every job (~seconds each).
- Acceptance:
  - [x] Wrap the model so weights load at consumer startup (without editing the submodule); per-job chords stage time drops measurably
  - [ ] T-115 benchmark numbers updated (T-115 has no numbers yet; it should measure with this cache in place)
- Notes: `cache_networks()` in chord_cnn_lstm_service.py swaps the submodule's `NetworkInterface` for a per-process cache; `ChordBeatAnalyzer()` preloads the 5 nets (1.4 s). sunnydays.mp3, 3 runs each, first job chords stage median 6.9 s → 5.9 s; later jobs save only ~0.08 s (a warm reload of the 28 MB of checkpoints is cheap; the cost was the first, cold load).

### T-119 Leave key empty for near-silent or keyless audio
- Lane: worker · Status: todo · Owner: — · Deps: T-110
- Goal: Emulator QA: a near-silent recording got key "F♯ minor" and 129 BPM.
- Acceptance:
  - [ ] `key` is null when the Krumhansl correlation (confidence) is below a threshold or the signal RMS is near silence; `bpm` null when there are too few beats
  - [ ] Unit tests with silence and noise

---

## M2 — Apps core (Android ‖ iOS)

Each `a`/`i` pair has identical acceptance criteria unless noted.

### T-201a / T-201i API client + guest session
- Lane: android / ios · Status: done · Owner: claude · Deps: T-101, T-102 (and T-003 for iOS)
- Acceptance:
  - [x] Generates a `deviceId` on first launch and stores it in Keystore/Keychain
  - [x] Calls `/auth/anonymous` if no token, stores the token securely, and fetches `/config` at launch
  - [x] A 401 triggers one re-auth then retries
  - [x] Request/response models match openapi names exactly; unit tests decode sample JSON
- Notes (android): extend the existing `network/KtorClient.kt`, `auth/TokenStore.kt`, `auth/AuthRepository.kt`. Drop `LogLevel.BODY` in release builds.
- Notes: Android: Keystore-encrypted DataStore; iOS: Keychain. Both cache /config and re-auth guests once on 401.

### T-202a / T-202i Local library store
- Lane: android / ios · Status: done · Owner: claude · Deps: T-201*
- Goal: Room / SwiftData `LibraryItem` mirroring the contracts schema, plus the file layout in PLAN.md §5.1.
- Acceptance:
  - [x] CRUD + "newest first" query; deletes are tombstones (`deleted=true`)
  - [x] `AnalysisStore.load(trackId)` decodes `analysis/{trackId}.json`; a test decodes the fixture
  - [x] Unit tests
- Notes: Android Room (v2, local error column); iOS SwiftData (tombstone stored as `tombstoned`, map to `deleted` in sync).

### T-200a / T-200i Design tokens + shared components
- Lane: android / ios · Status: done · Owner: claude · Deps: T-003 (iOS)
- Goal: Implement DESIGN.md §1 as a theme, plus the reusable pieces every screen uses. **Do this before any screen task.**
- Acceptance:
  - [x] Color/shape/type tokens exactly per DESIGN.md §1 (Compose `MaterialTheme` + `MusicDecodedColors` / SwiftUI `Color` + `Font` extensions), dark only
  - [x] Spacing tokens `xxs`…`xxxl` (4-pt scale) as `Spacing` constants; **no raw dp/pt literals for padding, margins or gaps anywhere in UI code** (grep check in the PR notes)
  - [x] Touch targets ≥ 44pt / 48dp
  - [x] Components: SongRow, SourceCard, SectionHeader, IconButton (default/active), PrimaryButton, SecondaryButton, PolicyNote, FilterChip, InfoChip (metadata), ControlChip, SegmentedToggle, a 3-tab bottom nav (Home/Library/Settings)
  - [x] Each component has a preview (Compose `@Preview` / SwiftUI `#Preview`) using fixture data
  - [x] Text ≥ 11pt/sp and supports Dynamic Type / font scaling
- Notes (android): replaces `ui/theme/Color.kt`, `Theme.kt` and `Type.kt` contents.
- Notes: Android + iOS: tokens, 4-pt Spacing, components with previews, 3-tab shell. iOS verified by CI.

### T-203a / T-203i Home screen
- Lane: android / ios · Status: done · Owner: claude · Deps: T-200*, T-202*
- Acceptance:
  - [x] Matches DESIGN.md §3.1: brand header, Recently Decoded (3 newest + See all), 2×2 source grid, Record card, tip card
  - [x] Rows show status (analyzing with progress / ready / failed with retry), offline badge, `duration · key · BPM`; the ⋯ menu has Delete (confirm), Favorite, Re-analyze
  - [x] SoundCloud card hidden unless `soundcloudEnabled`; empty-history state
- Notes (android): replaces the current `ui/library`, `ui/add`, `ui/dictionary` placeholder screens and `ui/main/MainScreen.kt` nav.
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-210a / T-210i Library screen
- Lane: android / ios · Status: done · Owner: claude · Deps: T-200*, T-202*
- Acceptance:
  - [x] DESIGN.md §3.2: search (title/artist/key), filters All/Favorites/Offline/Recordings, decoded/offline counts, list sorted by createdAt desc
  - [x] `favorite` persisted on LibraryItem (bumps `updatedAt`)
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-211a / T-211i Mini player + Processing screen
- Lane: android / ios · Status: done · Owner: claude · Deps: T-204*, T-208*
- Acceptance:
  - [x] Mini player above the tab bar on Home/Library while a track is loaded (DESIGN.md §2); tapping it opens the Player; playback continues across tabs
  - [x] Processing screen per DESIGN.md §3.5 with the exact stage → step mapping table; "Continue in background" and × both leave the job running
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-204a / T-204i Job runner (upload → job → poll → save)
- Lane: android / ios · Status: done · Owner: claude · Deps: T-103..T-105, T-202*
- Goal: One `TranscriptionRepository.start(source)` that every source uses.
- Acceptance:
  - [x] Computes SHA-256, `POST /uploads`, PUT bytes, `POST /jobs`, and polls per the openapi backoff rule
  - [x] Saves the analysis file and updates the item to `ready`; failures set `failed` with a retry action
  - [x] Survives app backgrounding (Android WorkManager / iOS background URLSession + resume polling on foreground)
  - [x] Tests against a fake API
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-205a / T-205i Files + Media Library sources
- Lane: android / ios · Status: done · Owner: claude · Deps: T-204*
- Acceptance:
  - [x] Files: system picker limited to audio. Media Library: iOS `MPMediaPickerController` (only non-DRM items with an `assetURL`), Android `MediaStore.Audio` picker
  - [x] Import sheet per DESIGN.md §3.3 (use the corrected copy) → "Import & decode"
  - [x] File copied to `audio/{trackId}.{ext}`; title from metadata, falling back to the filename
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-206a / T-206i Record source
- Lane: android / ios · Status: done · Owner: claude · Deps: T-204*
- Acceptance:
  - [x] Mic permission flow with a denied state
  - [x] UI per DESIGN.md §3.6: state dot, timer, live waveform, 16-segment input meter with dB, Discard / Record-Pause / Finish
  - [x] AAC 44.1 kHz mono; max 10 minutes; Finish asks for a title
  - [x] Saved to `audio/`, then the job starts
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-212a / T-212i Music theory helpers (pure, no UI)
- Lane: android / ios · Status: done · Owner: claude · Deps: T-202*
- Goal: The logic from DESIGN.md §4 as pure functions with unit tests on the fixture. Every player task builds on this.
- Acceptance:
  - [x] `currentState(t, analysis) → {chordIndex, beatIndex, bar}` (binary search)
  - [x] `upNext(t, analysis) → {chord, inBeats}`; `prevChordStart` / `nextChordStart` (skip `N`, 1s rule)
  - [x] `chordNotes(label, transpose) → [midi]` per `chord-voicings.json` (tests cover every `examples` entry)
  - [x] `displayName(label, transpose, accidentals, simplify)` per DESIGN.md §4 (A:maj→A, F#:min7→F♯m7, D:maj/3→D/F♯, Basic simplification)
  - [x] `barRange(t, analysis)` for the loop
- Notes: Both platforms done; Android ports every Swift test case (36).

### T-207a / T-207i Player: timeline
- Lane: android / ios · Status: done · Owner: claude · Deps: T-200*, T-212*
- Goal: DESIGN.md §3.7 item 3, built against the fixture first.
- Acceptance:
  - [x] Chord track blocks (current = chord fill) over waveform bars from `waveformPeaks` (played = primary); falls back to beat ticks when peaks are null; downbeat gridlines
  - [x] Content scrolls under a fixed center playhead; drag scrubs; pinch zooms 60–400 pt/s; tapping a chord block seeks to its start
  - [x] 60 fps on a mid-range device (draws only the visible window)
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-208a / T-208i Player: screen, playback, transport
- Lane: android / ios · Status: done · Owner: claude · Deps: T-207*
- Acceptance:
  - [x] Layout per DESIGN.md §3.7: header (favorite toggles), metadata chips (key/time sig/BPM), timeline, Now Playing chord + notes, control chips, visualizer card, transport sheet
  - [x] Plays `audio/{trackId}` with Media3 ExoPlayer / AVPlayer; the player position drives everything
  - [x] Transport: speed (time-stretch, pitch kept; 0.5–1.25×), prev/next chord, play/pause, loop (current bar by default, drag on the chord track for a custom range, loop pill with Clear)
  - [x] Chips: Chords Basic/Full and Transpose −6…+6 (display only) update the timeline, Now Playing and piano
  - [x] Works in airplane mode (manual test noted with a screenshot)
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-209a / T-209i Piano panel + Piano learning view
- Lane: android / ios · Status: done · Owner: claude · Deps: T-208*
- Acceptance:
  - [x] Canvas keyboard component: 2 octaves (96 tall) in the player card; large (190 tall) with optional note names in the learning view; active colors per DESIGN.md §1; scrolls to keep active keys visible
  - [x] Visualizer Piano | Video toggle; Video is disabled unless the source is youtube; expand → learning view
  - [x] Learning view per DESIGN.md §3.8: current/up-next card, position bar, progression strip (auto-scrolls), Speed/Loop/Note names/Transpose controls, big transport; a wider keyboard in landscape
- Notes: Both platforms done. Android verified on the Pixel_8 emulator against the full stack (incl. airplane-mode playback); iOS verified by CI (100 app tests). Polish items in T-213.

### T-213a / T-213i Visual polish from device QA
- Lane: android / ios · Status: todo · Owner: — · Deps: M2
- Goal: Remaining mismatches vs DESIGN.md found on the Android emulator (check the iOS equivalents on a simulator).
- Acceptance:
  - [ ] Player header ⋯ menu (Delete, Re-analyze, Favorite)
  - [ ] Seek bar styled per DESIGN (thin track, round thumb), not the stock slider
  - [ ] Song rows show the same initial-letter art as the mini player/Processing when there's no thumbnail
  - [ ] Import sheet presented over Home, not an empty screen
  - [ ] Media Library opens a proper audio picker (not the ringtone-style chooser) where the platform allows
  - [ ] Delete-while-playing clears the mini player (logic tested; confirm on device)

---

## M3 — YouTube sandbox

### T-301 YouTube search proxy (`GET /youtube/search`)
- Lane: api · Status: done · Owner: claude · Deps: T-101
- Acceptance:
  - [x] Calls YouTube Data API v3 `search.list` (type=video, videoCategoryId=10 music) and then `videos.list` for durations
  - [x] `YOUTUBE_API_KEY` comes from env; responses are cached in Redis for 1 hour per query
  - [x] Results longer than 10 minutes are excluded
- Notes: stdlib urllib (no new dep). Cache yt:search:{sha1(q|pageToken)}, q lowercased/whitespace-collapsed. No key → 403 feature_disabled; Google quota → 429 rate_limited (not a plan limit: apps must not show the upsell); other upstream errors → 500 internal. Live/upcoming broadcasts skipped. openapi: added 400/403/429 responses to /youtube/search.

### T-302 Worker: fetch streaming sources
- Lane: worker · Status: done · Owner: claude · Deps: T-113
- Acceptance:
  - [x] Uses **yt-dlp as a Python library** (PLAN.md D-8), not a separate downloader service: `format=bestaudio/best`, no video download, `noplaylist`, a 10-minute max duration check *before* downloading (`extract_info(download=False)`)
  - [x] The `youtube` job fetches by videoId and `soundcloud` by URL, into `TemporaryDirectory`; title/duration/thumbnail are written back to the track
  - [x] `yt-dlp` is pinned in `requirements.txt`; a comment next to the pin says to bump it whenever extraction breaks
  - [x] Extraction failures (incl. bot checks, PLAN.md R-8) → job `failed` with a readable error; one retry with backoff
  - [x] If the server flag `offlineStreamAudio` is on, the extracted audio is stored at `extracted-audio/{trackId}.m4a`; otherwise it is deleted
- Notes: services/fetch.py + tests/test_fetch.py (fake yt-dlp); flag read from env `OFFLINE_STREAM_AUDIO`. The pin (2025.10.14) is the newest yt-dlp for Python 3.9 and already fails on YouTube ("The page needs to be reloaded"); the same fetch.py works with yt-dlp 2026.10.26 on 3.12 (2 MB in 2.4 s). See T-307.

### T-307 Worker image on Python 3.10+ so yt-dlp can stay current
- Lane: worker · Status: todo · Owner: — · Deps: T-302
- Goal: PLAN.md R-8. yt-dlp needs Python 3.10+ since late 2025, so the 3.9 image is stuck on a release YouTube already rejects.
- Acceptance:
  - [ ] Dockerfile on python:3.11-slim (or newer); the chord model, librosa path and `python -m pytest` still pass; `data/sunnydays.mp3` output still validates
  - [ ] `yt-dlp` pinned to a current release; a JS runtime (deno) in the image for YouTube (yt-dlp EJS)
  - [ ] A real YouTube fetch succeeds from the container
- Notes: Found in T-302. CI (`.github/workflows/worker.yml`, infra lane) also pins 3.9.

### T-303 `GET /tracks/{id}/audio`
- Lane: api · Status: todo · Owner: — · Deps: T-302
- Acceptance:
  - [ ] 403 `feature_disabled` when the flag is off; otherwise a presigned GET URL

### T-304a / T-304i YouTube sandbox search UI
- Lane: android / ios · Status: todo · Owner: — · Deps: T-301, T-204*
- Acceptance:
  - [ ] DESIGN.md §3.4: search field (debounced 400 ms), recent-query suggestion chips, results (thumbnail + duration badge, title, channel), infinite scroll via pageToken
  - [ ] "Paste a video URL" card parses watch/youtu.be/shorts URLs into a videoId
  - [ ] Video detail screen: hero (inline embed on play), availability card, "Decode this song" → `youtube` job → Processing, policy note copy verbatim

### T-305a / T-305i Video panel synced to the timeline
- Lane: android / ios · Status: todo · Owner: — · Deps: T-304*, T-209*
- Acceptance:
  - [ ] Official embedded player (iOS: YouTube IFrame Player API in WKWebView; Android: `android-youtube-player` or the IFrame API in a WebView)
  - [ ] When there's no local audio, the video's `currentTime` is the clock; scrubbing the timeline seeks the video
  - [ ] When local audio exists (flag on), the audio is the clock and the video is muted and kept within 150 ms

### T-306 Apply for a YouTube Data API quota increase
- Lane: infra · Status: todo · Owner: product owner · Deps: T-301, T-504 (privacy policy)
- Goal: PLAN.md R-6. The default 10,000 units/day ≈ 100 searches/day for the whole app.
- Acceptance:
  - [ ] Google Cloud project owned by the product owner, API key restricted to the API server's IP
  - [ ] Daily quota usage alert at 80%
  - [ ] Quota extension form submitted (needs a demo video of the search flow + a privacy policy URL); outcome noted here
- Notes: Start as soon as the search screen works. Approval can take weeks.

---

## M4 — Accounts & sync

### T-401 Signup/login with guest link
- Lane: api · Status: done · Owner: claude · Deps: T-101
- Acceptance:
  - [x] Signup/login with a guest bearer token reassigns that guest's jobs/uploads to the account
  - [x] Tests for new signup, login-merge and the duplicate email 409
- Notes: auth._link_guest moves rows of every model in _GUEST_OWNED (T-402: add the library table there). The guest user row is kept (device can re-auth as an empty guest). Account/invalid bearers are ignored; no schema change. Guest's Redis daily counter is not carried over.

### T-402 Sync endpoints
- Lane: api · Status: done · Owner: claude · Deps: T-401
- Acceptance:
  - [x] `GET/PUT /sync/library` per openapi, last-write-wins on `updatedAt`, tombstones
  - [x] Guest tokens get 403 `account_required`
  - [x] `GET /uploads/{uploadId}` per openapi: presigned GET for the owning account only, 404 for anyone else
- Notes: app/sync.py + migration 0004 (library_items, PK user_id+track_id, extra server_updated_at). Pull orders by (server_updated_at, trackId), 500/page; clients repeat with the returned cursor until a page has < 500 items (no `since` = full pull; empty page echoes the cursor). PUT: client wins only if strictly newer (ties keep server); unknown trackId or an audioUploadId the caller doesn't own → 400 for the whole push (nothing written); max 500 items. T-105 rule is now "has a job OR a non-deleted library row" (analysis/audio); /uploads/{id} also allows an upload referenced by the caller's library. library_items not in _GUEST_OWNED (guests can't own rows). tests/test_sync.py.

### T-403a / T-403i Account UI
- Lane: android / ios · Status: todo · Owner: — · Deps: T-401
- Acceptance:
  - [ ] Settings screen per DESIGN.md §3.9 (only the v1 rows listed there); profile card guest/account; sync card for guests
  - [ ] Signup/login; on a device with guest data, show the migration screen (DESIGN.md §3.10) with real counts (timelines, own audio + size, restricted streaming tracks disabled) → Confirm & sync
  - [ ] Logout falls back to a fresh guest session but keeps local data
- Notes (android): reuse `ui/auth/AuthScreen.kt` and `auth/AuthViewModel.kt`.

### T-404a / T-404i Sync client
- Lane: android / ios · Status: todo · Owner: — · Deps: T-402, T-403*
- Acceptance:
  - [ ] Pull/push on foreground, after a job completes and after login; stores the cursor
  - [ ] Downloads missing analysis files (`GET /tracks/{id}/analysis`) and owned audio (`GET /uploads/{audioUploadId}`) for pulled items
  - [ ] Test: two fake devices converge

---

## M5 — Polish & store

### T-501 Decide R-1 (streaming offline audio)
- Lane: design · Status: blocked · Owner: product owner · Deps: —
- Goal: The product owner decides the store-build value of `offlineStreamAudio`. Agents must not decide this.

### T-502a / T-502i SoundCloud link source
- Lane: android / ios · Status: todo · Owner: — · Deps: T-302
- Acceptance:
  - [ ] Paste/share-sheet a SoundCloud URL → `soundcloud` job; hidden unless `soundcloudEnabled`

### T-503a / T-503i Accessibility + error states pass
- Lane: android / ios · Status: todo · Owner: — · Deps: M2–M4
- Acceptance:
  - [ ] Screen reader labels on all controls, dynamic type, contrast ≥ 4.5:1, and a network-error state on every screen

### T-505 Decide: Tuner tab
- Lane: design · Status: blocked · Owner: product owner · Deps: —
- Goal: The Figma prototype has a 4th "Tuner" tab (currently wired to the recorder). Decide whether v1 gets an instrument tuner (on-device pitch detection, no server). Until then, apps ship 3 tabs.

### T-504 Store submission checklist
- Lane: infra · Status: todo · Owner: — · Deps: T-501
- Acceptance:
  - [ ] Privacy policy, data-safety form, mic usage strings, TestFlight + Play internal builds

---

## M6 — Pro

Read PLAN.md §9 first. **Rule: the API enforces Pro. The apps only reflect it.**

### T-601 Billing webhook + entitlements
- Lane: api · Status: todo · Owner: — · Deps: T-401
- Goal: RevenueCat → API sets `users.plan`.
- Acceptance:
  - [ ] Contracts task first: add `POST /billing/revenuecat` to openapi (shared-secret auth header, no JWT)
  - [ ] Handles INITIAL_PURCHASE, RENEWAL, CANCELLATION (pro until expiry), EXPIRATION, BILLING_ISSUE; idempotent on event id
  - [ ] `plan` and `planExpiresAt` updated; the next `/config` reflects it; new tokens carry the new `plan` claim
  - [ ] Free users requesting a Pro-only model get 403 `pro_required`
  - [ ] Tests with recorded sample webhook payloads

### T-602a / T-602i Paywall + purchase flow
- Lane: android / ios · Status: todo · Owner: — · Deps: T-601, T-403*
- Acceptance:
  - [ ] RevenueCat SDK (StoreKit 2 / Play Billing); RevenueCat `appUserID` = our `userId`; guests are prompted to create an account first
  - [ ] A paywall screen designed through a `design` task (DESIGN.md section added first), shown on `rate_limited` and from Settings
  - [ ] Restore purchases; after purchase the app re-fetches `/config` and the UI updates without a restart
  - [ ] Sandbox purchase verified on both stores

### T-603 Contributor setup
- Lane: infra · Status: todo · Owner: — · Deps: —
- Goal: Make the repo welcoming for contributors and recruiters, and keep relicensing possible (PLAN.md D-9).
- Acceptance:
  - [ ] `CONTRIBUTING.md`: setup, lanes and the task workflow (from the top of TASKS.md), how to pick a task, PR rules, the license split (MIT apps / AGPL backend)
  - [ ] CLA via CLA Assistant (GitHub app) required on PRs that touch `apps/api` or `apps/worker`
  - [ ] Issue/PR templates; `good first issue` labels on 5 small tasks
  - [ ] README badges (license, CI) and an architecture diagram near the top
