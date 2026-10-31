# User Flows

Screens and visuals: [`DESIGN.md`](../DESIGN.md). Scope: [`PLAN.md`](../PLAN.md) §3.

## First launch
1. Generate `deviceId` → store in Keychain/Keystore.
2. `POST /auth/anonymous` → guest token. `GET /config` → flags.
3. Show Home (empty state). No sign-up prompt.

## Transcribe: Files / Record
1. Home → Transcribe → Files (picker) or Record (recorder).
2. Copy/save audio to `audio/{tempId}`; create LibraryItem `status=analyzing`.
3. SHA-256 → `POST /uploads` → PUT → `POST /jobs {source, uploadId, title}`.
4. Rename local files to the returned `trackId`. Poll `GET /jobs/{id}`; the history row shows progress.
5. On `complete`: `GET /tracks/{id}/analysis` → save `analysis/{trackId}.json` → `status=ready`.
6. On `failed`: `status=failed`, row shows Retry (re-runs from step 3, reusing the upload).

## Transcribe: YouTube sandbox
1. Home → Transcribe → YouTube → search screen.
2. Type a query → `GET /youtube/search` → results.
3. Tap a result → `POST /jobs {source: youtube, videoId}` → LibraryItem with thumbnail, title, `status=analyzing` → back to Home.
4. Same polling/saving as above.
5. If `offlineStreamAudio` is true: also `GET /tracks/{id}/audio` → save to `audio/`.

## Transcribe: SoundCloud (M5)
Paste or share a URL → `POST /jobs {source: soundcloud, soundcloudUrl}` → same as YouTube.

## Play
1. Tap a `ready` row → Player loads `analysis/{trackId}.json` (no network needed).
2. Clock = local audio if present; else the YouTube player (network needed).
3. Each frame: `currentState(t, analysis)` → highlight chord, beat, piano notes.

## Create account / sync (optional)
1. Settings → "Sync across devices" → sign up or log in (the request carries the guest token).
2. The server reassigns guest data to the account; the app swaps in the account token.
3. Sync runs: `PUT /sync/library` (push everything local), then `GET /sync/library`.
4. On another device: log in → pull → for each item missing locally, download the analysis (and owned audio).
5. Log out: local data stays; the app gets a fresh guest session.

## Delete
Swipe/long-press → confirm → `deleted=true`, `updatedAt=now`, local files removed. Pushed on next sync.
