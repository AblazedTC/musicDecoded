# TASKS.md — Work board

Rules: see `AGENTS.md` §2. Pick the lowest-numbered `todo` task in your lane whose deps are all `done`.
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
- Lane: ios · Status: todo · Owner: — · Deps: T-001
- Goal: A SwiftUI app `MusicDecoded` in `apps/ios` with app + unit test targets, iOS 17, bundle id `com.musicdecoded.app`.
- Acceptance:
  - [ ] `xcodebuild … build` and `… test` succeed on the simulator
  - [ ] Folder layout matches `apps/ios/AGENTS.md`
  - [ ] `contracts/fixtures/sample.analysis.json` and `contracts/chord-voicings.json` are added to the test bundle **by reference** (not copied)
  - [ ] Tab/nav shell with an empty Home screen
- Notes: Requires macOS.

### T-004 Remove the hardcoded JWT secret fallback
- Lane: api · Status: todo · Owner: — · Deps: —
- Goal: The app refuses to start without `JWT_SECRET` unless the `dev` profile is active.
- Acceptance:
  - [ ] `application.properties` has `jwt.secret=${JWT_SECRET}`; the dev default lives only in `application-dev.properties`
  - [ ] A test shows that context startup fails without the secret in the default profile
  - [ ] `docker-compose.yml` and `docs/running-locally.md` set `SPRING_PROFILES_ACTIVE=dev` for local use

### T-005 CI per lane
- Lane: infra · Status: todo · Owner: — · Deps: T-003
- Goal: GitHub Actions workflows filtered by path: contracts (validate + lint), api (mvn test), worker (pytest), android (assembleDebug + unit tests), ios (macos runner, xcodebuild test).
- Acceptance:
  - [ ] Each workflow runs only when its lane's paths change
  - [ ] A PR touching `contracts/**` runs **all** workflows (contract changes can break every lane)

### T-006 Local infrastructure: Redis + MinIO
- Lane: infra · Status: todo · Owner: — · Deps: —
- Goal: Add `redis` and `minio` (plus a bucket-init step) to `docker-compose.yml`, and wire their env vars into the api and worker.
- Acceptance:
  - [ ] `docker compose up -d` brings up db, redis, minio and a `musicdecoded` bucket
  - [ ] The env vars are documented in `docs/running-locally.md`

### T-007 Worker repo hygiene
- Lane: worker · Status: todo · Owner: — · Deps: —
- Goal: Remove committed `__pycache__`, gitignore `outputs/` and `data/*.mp3` (keep one small test clip), and add `pytest` with one smoke test.
- Acceptance:
  - [ ] `git ls-files apps/worker | grep pycache` is empty
  - [ ] `python -m pytest` runs and passes

---

## M1 — Pipeline

### T-101 Guest auth (`POST /auth/anonymous`)
- Lane: api · Status: todo · Owner: — · Deps: T-004
- Goal: Issue guest JWTs keyed by deviceId, and add the `kind` and `plan` claims to every token.
- Acceptance:
  - [ ] Same deviceId → same userId (idempotent)
  - [ ] `User` gains `kind` (guest/account), `plan` (free/pro, default free), `planExpiresAt`, a nullable email/password, and `deviceId`
  - [ ] Existing signup/login return `AuthResponse` per openapi (token, userId, kind, plan)
  - [ ] Tests for both paths
- Notes: Reuse `auth/JwtService`, `AuthService`, `SecurityConfig`.

### T-102 `GET /config` and `/health`
- Lane: api · Status: todo · Owner: — · Deps: —
- Acceptance:
  - [ ] Flags come from env vars (`OFFLINE_STREAM_AUDIO`, default false; `SOUNDCLOUD_ENABLED`, default false)
  - [ ] Returns the caller's `plan` and `limits` (`dailyJobs` from the plan's limit, `jobsUsedToday` from the Redis counter)
  - [ ] Matches the `ClientConfig` schema

### T-103 Uploads (`POST /uploads`)
- Lane: api · Status: todo · Owner: — · Deps: T-006, T-101
- Goal: Validate content type and size, then return a presigned S3/MinIO PUT URL at `uploads/{uploadId}`.
- Acceptance:
  - [ ] Rejects >100 MB with 413 `too_large`
  - [ ] Stores `{uploadId, userId, sha256, contentType}`
  - [ ] An integration test PUTs a file to the URL against MinIO

### T-104 Jobs + tracks (`POST /jobs`, `GET /jobs/{id}`)
- Lane: api · Status: todo · Owner: — · Deps: T-103
- Goal: Create or reuse a track by dedupe key (sha256 / videoId / soundcloudUrl), create a job, and enqueue it to Redis list `jobs:analysis`.
- Acceptance:
  - [ ] An already-analyzed dedupe key returns 200 `complete` without enqueueing
  - [ ] Queue message format is documented in `docs/data-model.md`
  - [ ] Per-user daily limit by plan, from config (`LIMIT_GUEST=10`, `LIMIT_FREE=50`, `LIMIT_PRO=` empty = unlimited) → 429 `rate_limited`. Cache hits (already analyzed) don't count
  - [ ] `GET /jobs/{id}` returns 404 for another user's job

### T-105 Analysis download (`GET /tracks/{id}/analysis`)
- Lane: api · Status: todo · Owner: — · Deps: T-104
- Acceptance:
  - [ ] Returns a presigned GET URL for `analysis/{trackId}.json` (5-minute expiry)
  - [ ] 404 until the job is complete

### T-110 Worker v1 output (key, BPM, time signature)
- Lane: worker · Status: todo · Owner: — · Deps: T-002, T-007
- Goal: Extend `ChordBeatAnalyzer` (`services/chord_beat_analysis.py`) to emit analysis v1 instead of the legacy format.
- Acceptance:
  - [ ] Output for `data/sunnydays.mp3` passes `python contracts/validate.py <out>`
  - [ ] Key via Krumhansl-Schmuckler on mean chroma; BPM = 60 / median IBI; time signature = median beats per bar
  - [ ] Unit tests for key/BPM/time-signature on synthetic inputs

### T-111 Worker spectrogram + waveform peaks
- Lane: worker · Status: todo · Owner: — · Deps: T-110
- Acceptance:
  - [ ] 64-bin mel, 20 fps, dB clipped to [-80, 0] → uint8 → base64; peaks at 20 fps normalized to 0–1
  - [ ] A 4-minute song produces an analysis file < 1.5 MB

### T-112 Regenerate the fixture from real worker output
- Lane: contracts · Status: todo · Owner: — · Deps: T-111
- Acceptance:
  - [ ] `contracts/fixtures/sample.analysis.json` has a real key, spectrogram and peaks, and validates

### T-113 Worker queue consumer
- Lane: worker · Status: todo · Owner: — · Deps: T-104, T-110
- Goal: A long-running process that BLPOPs `jobs:analysis`, downloads the upload from storage, runs the analysis, writes `analysis/{trackId}.json`, and updates the job's status/stage/progress in Postgres and Redis.
- Acceptance:
  - [ ] Stages match the openapi `Job.stage` enum
  - [ ] On exception: status `failed` with an error message, and temp files deleted
  - [ ] Temp media is deleted on success too
  - [ ] The Dockerfile `CMD` runs the consumer

### T-114 End-to-end pipeline script
- Lane: infra · Status: todo · Owner: — · Deps: T-105, T-113
- Goal: `scripts/e2e.sh`: guest auth → upload an mp3 → job → poll → download → validate.
- Acceptance:
  - [ ] Passes against `docker compose up` from a clean checkout

### T-115 Benchmark analysis cost
- Lane: worker · Status: todo · Owner: — · Deps: T-111
- Goal: Real numbers for PLAN.md §9's cost model, before any pricing decision.
- Acceptance:
  - [ ] The worker logs wall time per stage (fetch, normalize, beats, chords, key, spectrogram, save) for every job
  - [ ] `scripts/benchmark.py` runs 20 varied songs (2–8 min, different genres) on the target CPU instance size and reports median/p90 seconds per song, per stage, and peak RAM
  - [ ] Results + `$ per 1,000 uncached songs` for 2 candidate hosts (e.g. Fly.io/Railway CPU) written to `docs/cost-model.md`
  - [ ] The same run on a GPU host if Beat Transformer is enabled (the Pro model path)

### T-116 Remove the madmom fallback (non-commercial model license)
- Lane: worker · Status: todo · Owner: — · Deps: T-007
- Goal: PLAN.md R-7. madmom's pretrained models are CC BY-NC-SA and can't ship in a product with paid plans.
- Acceptance:
  - [ ] `ChordBeatAnalyzer` no longer imports or calls `MadmomDetectorService`; `madmom` removed from `requirements.txt` and the Dockerfile
  - [ ] librosa beat tracking is the free-tier path; Beat Transformer (MIT) is the premium path
  - [ ] `apps/worker/MODELS.md` lists every model and its **code and weights** licenses (Beat Transformer: MIT; ISMIR2019 chord model: MIT; librosa: ISC)
  - [ ] The output for `data/sunnydays.mp3` still validates, and beat count is within ±5% of before

---

## M2 — Apps core (Android ‖ iOS)

Each `a`/`i` pair has identical acceptance criteria unless noted.

### T-201a / T-201i API client + guest session
- Lane: android / ios · Status: todo · Owner: — · Deps: T-101, T-102 (and T-003 for iOS)
- Acceptance:
  - [ ] Generates a `deviceId` on first launch and stores it in Keystore/Keychain
  - [ ] Calls `/auth/anonymous` if no token, stores the token securely, and fetches `/config` at launch
  - [ ] A 401 triggers one re-auth then retries
  - [ ] Request/response models match openapi names exactly; unit tests decode sample JSON
- Notes (android): extend the existing `network/KtorClient.kt`, `auth/TokenStore.kt`, `auth/AuthRepository.kt`. Drop `LogLevel.BODY` in release builds.

### T-202a / T-202i Local library store
- Lane: android / ios · Status: todo · Owner: — · Deps: T-201*
- Goal: Room / SwiftData `LibraryItem` mirroring the contracts schema, plus the file layout in PLAN.md §5.1.
- Acceptance:
  - [ ] CRUD + "newest first" query; deletes are tombstones (`deleted=true`)
  - [ ] `AnalysisStore.load(trackId)` decodes `analysis/{trackId}.json`; a test decodes the fixture
  - [ ] Unit tests

### T-200a / T-200i Design tokens + shared components
- Lane: android / ios · Status: todo · Owner: — · Deps: T-003 (iOS)
- Goal: Implement DESIGN.md §1 as a theme, plus the reusable pieces every screen uses. **Do this before any screen task.**
- Acceptance:
  - [ ] Color/shape/type tokens exactly per DESIGN.md §1 (Compose `MaterialTheme` + `MusicDecodedColors` / SwiftUI `Color` + `Font` extensions), dark only
  - [ ] Spacing tokens `xxs`…`xxxl` (4-pt scale) as `Spacing` constants; **no raw dp/pt literals for padding, margins or gaps anywhere in UI code** (grep check in the PR notes)
  - [ ] Touch targets ≥ 44pt / 48dp
  - [ ] Components: SongRow, SourceCard, SectionHeader, IconButton (default/active), PrimaryButton, SecondaryButton, PolicyNote, FilterChip, InfoChip (metadata), ControlChip, SegmentedToggle, a 3-tab bottom nav (Home/Library/Settings)
  - [ ] Each component has a preview (Compose `@Preview` / SwiftUI `#Preview`) using fixture data
  - [ ] Text ≥ 11pt/sp and supports Dynamic Type / font scaling
- Notes (android): replaces `ui/theme/Color.kt`, `Theme.kt` and `Type.kt` contents.

### T-203a / T-203i Home screen
- Lane: android / ios · Status: todo · Owner: — · Deps: T-200*, T-202*
- Acceptance:
  - [ ] Matches DESIGN.md §3.1: brand header, Recently Decoded (3 newest + See all), 2×2 source grid, Record card, tip card
  - [ ] Rows show status (analyzing with progress / ready / failed with retry), offline badge, `duration · key · BPM`; the ⋯ menu has Delete (confirm), Favorite, Re-analyze
  - [ ] SoundCloud card hidden unless `soundcloudEnabled`; empty-history state
- Notes (android): replaces the current `ui/library`, `ui/add`, `ui/dictionary` placeholder screens and `ui/main/MainScreen.kt` nav.

### T-210a / T-210i Library screen
- Lane: android / ios · Status: todo · Owner: — · Deps: T-200*, T-202*
- Acceptance:
  - [ ] DESIGN.md §3.2: search (title/artist/key), filters All/Favorites/Offline/Recordings, decoded/offline counts, list sorted by createdAt desc
  - [ ] `favorite` persisted on LibraryItem (bumps `updatedAt`)

### T-211a / T-211i Mini player + Processing screen
- Lane: android / ios · Status: todo · Owner: — · Deps: T-204*, T-208*
- Acceptance:
  - [ ] Mini player above the tab bar on Home/Library while a track is loaded (DESIGN.md §2); tapping it opens the Player; playback continues across tabs
  - [ ] Processing screen per DESIGN.md §3.5 with the exact stage → step mapping table; "Continue in background" and × both leave the job running

### T-204a / T-204i Job runner (upload → job → poll → save)
- Lane: android / ios · Status: todo · Owner: — · Deps: T-103..T-105, T-202*
- Goal: One `TranscriptionRepository.start(source)` that every source uses.
- Acceptance:
  - [ ] Computes SHA-256, `POST /uploads`, PUT bytes, `POST /jobs`, and polls per the openapi backoff rule
  - [ ] Saves the analysis file and updates the item to `ready`; failures set `failed` with a retry action
  - [ ] Survives app backgrounding (Android WorkManager / iOS background URLSession + resume polling on foreground)
  - [ ] Tests against a fake API

### T-205a / T-205i Files + Media Library sources
- Lane: android / ios · Status: todo · Owner: — · Deps: T-204*
- Acceptance:
  - [ ] Files: system picker limited to audio. Media Library: iOS `MPMediaPickerController` (only non-DRM items with an `assetURL`), Android `MediaStore.Audio` picker
  - [ ] Import sheet per DESIGN.md §3.3 (use the corrected copy) → "Import & decode"
  - [ ] File copied to `audio/{trackId}.{ext}`; title from metadata, falling back to the filename

### T-206a / T-206i Record source
- Lane: android / ios · Status: todo · Owner: — · Deps: T-204*
- Acceptance:
  - [ ] Mic permission flow with a denied state
  - [ ] UI per DESIGN.md §3.6: state dot, timer, live waveform, 16-segment input meter with dB, Discard / Record-Pause / Finish
  - [ ] AAC 44.1 kHz mono; max 10 minutes; Finish asks for a title
  - [ ] Saved to `audio/`, then the job starts

### T-212a / T-212i Music theory helpers (pure, no UI)
- Lane: android / ios · Status: todo · Owner: — · Deps: T-202*
- Goal: The logic from DESIGN.md §4 as pure functions with unit tests on the fixture. Every player task builds on this.
- Acceptance:
  - [ ] `currentState(t, analysis) → {chordIndex, beatIndex, bar}` (binary search)
  - [ ] `upNext(t, analysis) → {chord, inBeats}`; `prevChordStart` / `nextChordStart` (skip `N`, 1s rule)
  - [ ] `chordNotes(label, transpose) → [midi]` per `chord-voicings.json` (tests cover every `examples` entry)
  - [ ] `displayName(label, transpose, accidentals, simplify)` per DESIGN.md §4 (A:maj→A, F#:min7→F♯m7, D:maj/3→D/F♯, Basic simplification)
  - [ ] `barRange(t, analysis)` for the loop

### T-207a / T-207i Player: timeline
- Lane: android / ios · Status: todo · Owner: — · Deps: T-200*, T-212*
- Goal: DESIGN.md §3.7 item 3, built against the fixture first.
- Acceptance:
  - [ ] Chord track blocks (current = chord fill) over waveform bars from `waveformPeaks` (played = primary); falls back to beat ticks when peaks are null; downbeat gridlines
  - [ ] Content scrolls under a fixed center playhead; drag scrubs; pinch zooms 60–400 pt/s; tapping a chord block seeks to its start
  - [ ] 60 fps on a mid-range device (draws only the visible window)

### T-208a / T-208i Player: screen, playback, transport
- Lane: android / ios · Status: todo · Owner: — · Deps: T-207*
- Acceptance:
  - [ ] Layout per DESIGN.md §3.7: header (favorite toggles), metadata chips (key/time sig/BPM), timeline, Now Playing chord + notes, control chips, visualizer card, transport sheet
  - [ ] Plays `audio/{trackId}` with Media3 ExoPlayer / AVPlayer; the player position drives everything
  - [ ] Transport: speed (time-stretch, pitch kept; 0.5–1.25×), prev/next chord, play/pause, loop (current bar by default, drag on the chord track for a custom range, loop pill with Clear)
  - [ ] Chips: Chords Basic/Full and Transpose −6…+6 (display only) update the timeline, Now Playing and piano
  - [ ] Works in airplane mode (manual test noted with a screenshot)

### T-209a / T-209i Piano panel + Piano learning view
- Lane: android / ios · Status: todo · Owner: — · Deps: T-208*
- Acceptance:
  - [ ] Canvas keyboard component: 2 octaves (96 tall) in the player card; large (190 tall) with optional note names in the learning view; active colors per DESIGN.md §1; scrolls to keep active keys visible
  - [ ] Visualizer Piano | Video toggle; Video is disabled unless the source is youtube; expand → learning view
  - [ ] Learning view per DESIGN.md §3.8: current/up-next card, position bar, progression strip (auto-scrolls), Speed/Loop/Note names/Transpose controls, big transport; a wider keyboard in landscape

---

## M3 — YouTube sandbox

### T-301 YouTube search proxy (`GET /youtube/search`)
- Lane: api · Status: todo · Owner: — · Deps: T-101
- Acceptance:
  - [ ] Calls YouTube Data API v3 `search.list` (type=video, videoCategoryId=10 music) and then `videos.list` for durations
  - [ ] `YOUTUBE_API_KEY` comes from env; responses are cached in Redis for 1 hour per query
  - [ ] Results longer than 10 minutes are excluded

### T-302 Worker: fetch streaming sources
- Lane: worker · Status: todo · Owner: — · Deps: T-113
- Acceptance:
  - [ ] Uses **yt-dlp as a Python library** (PLAN.md D-8), not a separate downloader service: `format=bestaudio/best`, no video download, `noplaylist`, a 10-minute max duration check *before* downloading (`extract_info(download=False)`)
  - [ ] The `youtube` job fetches by videoId and `soundcloud` by URL, into `TemporaryDirectory`; title/duration/thumbnail are written back to the track
  - [ ] `yt-dlp` is pinned in `requirements.txt`; a note in `apps/worker/AGENTS.md` says to bump it whenever extraction breaks
  - [ ] Extraction failures (incl. bot checks, PLAN.md R-8) → job `failed` with a readable error; one retry with backoff
  - [ ] If the server flag `offlineStreamAudio` is on, the extracted audio is stored at `extracted-audio/{trackId}.m4a`; otherwise it is deleted
- Notes: Read PLAN.md R-1 and R-8 first.

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
- Lane: api · Status: todo · Owner: — · Deps: T-101
- Acceptance:
  - [ ] Signup/login with a guest bearer token reassigns that guest's jobs/uploads to the account
  - [ ] Tests for new signup, login-merge and the duplicate email 409

### T-402 Sync endpoints
- Lane: api · Status: todo · Owner: — · Deps: T-401
- Acceptance:
  - [ ] `GET/PUT /sync/library` per openapi, last-write-wins on `updatedAt`, tombstones
  - [ ] Guest tokens get 403 `account_required`
  - [ ] An upload referenced by a synced item is readable by the same account on any device

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
  - [ ] Downloads missing analysis files and owned audio for pulled items
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
  - [ ] `CONTRIBUTING.md`: setup, lanes (link to AGENTS.md), how to pick a task, PR rules, the license split (MIT apps / AGPL backend)
  - [ ] CLA via CLA Assistant (GitHub app) required on PRs that touch `apps/api` or `apps/worker`
  - [ ] Issue/PR templates; `good first issue` labels on 5 small tasks
  - [ ] README badges (license, CI) and an architecture diagram near the top
