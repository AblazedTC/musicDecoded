# DESIGN.md — Screens, tokens & player spec

> **Source of truth for visuals**: Figma Make file **"Music Transcription Learning App"**
> https://www.figma.com/make/EEN7dqUVB7LlQFTNiFKvHz/Music-Transcription-Learning-App
> It's a React prototype: `src/App.tsx` (screens) and `src/index.css` (tokens). Read it through the Figma MCP (`get_design_context`, fileKey `EEN7dqUVB7LlQFTNiFKvHz`, nodeId `0:1`), then the `src/*` resources.
> This doc translates it for native SwiftUI / Compose. **The prototype is a reference, not code to port.** Rebuild with native components and layout. Prototype sample data (DAY6 "Somehow", Unsplash art, "Alex Morgan") is placeholder only.

Both apps implement the same screens and behavior. Platform conventions (navigation, sheets, back gestures, haptics) follow each OS.

---

## 1. Design tokens

**The app is dark-only in v1** (the prototype has no light theme; the Settings "Appearance" row is post-v1).

### Colors
| Token | Hex | Use |
|---|---|---|
| `ink` | `#07101C` | window background behind everything |
| `canvas` | `#0B1422` | screen background |
| `surface` | `#111F30` | cards, rows, chips |
| `surfaceRaised` | `#182A3E` | pressed/hover, gradient top of source cards |
| `surfaceSoft` | `#20354B` | selected segment, progress track, avatar bg |
| `line` | `#2A4057` | 1px borders |
| `primary` | `#62DCC4` | accent: playhead, play button, selected nav, progress, links |
| `onPrimary` | `#08221E` | text/icons on primary |
| `primarySoft` | `#173D3D` | tinted backgrounds (offline badge, sync card, done steps) |
| `secondary` | `#91AFFF` | secondary accent (media library, avatar icon, link icons) |
| `chord` | `#F2DFAF` | chord names, current chord fill, loop pill |
| `onChord` | `#322B1D` | text on chord fill |
| `chordMuted` | bg `#463F32`, border `#645943` | inactive chord blocks on the timeline |
| `text` | `#F5F8FB` | primary text |
| `textSecondary` | `#9FB0C3` | secondary text |
| `textMuted` | `#8396AA` | captions, inactive nav (≥ 4.5:1 on canvas, surface and transport; was #63778C, which failed WCAG AA) |
| `record` | `#EC6478` (ring `#B94D5E`) | record button |
| `transportBg` | `#152638` (border `#30465D`) | transport sheet, import sheet, mini player `#1B2D40` |
| `timelineBg` | `#09101A` (border `#24384B`) | waveform container |
| `waveIdle` | `#4E6379` @72% | unplayed waveform bars (played = `primary` @80%) |

Source-icon tints (icon / bg): Files `primary`/`primarySoft` · Media Library `secondary`/`#1D3153` · YouTube `#FF8B8B`/`#452534` · SoundCloud `#FFB56F`/`#432F26` · Record `#FF9A9A`/`#412534`.

Piano: white key `#E7EBEC` (border `#8D999D`), black key `#0C1218`, active white `#9DE8DB`, active black `#48BEA8`, keyboard bg `#0A111A`.

### Shape & elevation
| Token | Value |
|---|---|
| `radiusCard` | 18 (cards, timeline, visualizer) |
| `radiusRow` | 15 (song rows, mini player, summary) |
| `radiusControl` | 12 (search, buttons 13, chips 10) |
| `radiusPill` | 20 (filter chips) |
| `radiusSheet` | 22–24 top corners (transport, import sheet) |
| `shadowCard` | y12 blur30 black 20% |
| `shadowPlayer` | y−14 blur35 black 28% |

### Spacing
**4-point scale. Every margin, padding and gap uses one of these values and nothing else.**

| Token | `xxs` | `xs` | `sm` | `md` | `lg` | `xl` | `xxl` | `xxxl` |
|---|---|---|---|---|---|---|---|---|
| pt/dp | 4 | 8 | 12 | 16 | 20 | 24 | 32 | 40 |

The prototype uses off-grid values. **Round to the nearest token; on a tie, round down.** Common cases:

| Prototype | → Token | Where |
|---|---|---|
| 1.5–5 | `xxs` 4 | chord-block inset, icon-to-label gaps |
| 6–10 | `xs` 8 | list gaps, chip gaps, metadata row gap, card inner padding (10) |
| 11–14 | `sm` 12 | song-row gap, policy-note padding, icon tile margin (14), section header bottom |
| 15–18 | `md` 16 | **screen side gutter** (prototype 18), card padding, transport padding |
| 19–22 | `lg` 20 | section spacing (prototype 20–22) |
| 26–32 | `xl`/`xxl` | major block separation (26 → 24, 30–32 → 32) |
| 36–48 | `xxxl` 40 | processing/recorder top padding, record-actions gap (48 → 40) |

Fixed **component sizes** (heights, icon tiles, buttons) aren't spacing and keep the prototype values, rounded to whole points: icon button 40, small icon button 32, play button 56 (prototype 58), record button 84, song art 56, source icon tile 44 (prototype 43), primary button height 48 (prototype 50), search field 44 (prototype 43), tab bar = platform default. **Minimum touch target 44pt (iOS) / 48dp (Android)**, so pad smaller visuals up to it.

### Typography
The prototype uses Inter, falling back to SF Pro. **Native: use the system font** (SF Pro / Roboto) and map onto platform text styles so Dynamic Type and font scaling work.
The prototype's sizes are tuned for a ~400px web frame and some go as small as 8px. **Native minimum is 11pt/sp.** Scale up the prototype's 8–10px captions to the platform's `caption2`/`labelSmall`.

| Role | Prototype | iOS | Android |
|---|---|---|---|
| Page title | 28 / -0.05em, bold | `.largeTitle` bold | `headlineMedium` |
| Current chord (player) | 34, chord color | `.system(size: 34, weight: .bold)` | 34sp bold |
| Section header | 18 bold | `.title3` bold | `titleMedium` |
| Row title | 14 semibold | `.subheadline` semibold | `titleSmall` |
| Body / secondary | 12 | `.footnote` | `bodySmall` |
| Caption / overline | 8–10, often uppercase +0.1em tracking | `.caption2` | `labelSmall` |
| Times / counters | tabular numbers | `.monospacedDigit()` | `fontFeatureSettings = "tnum"` |

### Icons
Line icons, 24 grid, 1.8 stroke, round caps (custom set in `App.tsx` `Icon`). Native: **SF Symbols** on iOS and **Material Symbols Rounded** on Android with the closest match. Exact SVG parity isn't required. YouTube and SoundCloud marks must follow their brand guidelines.

---

## 2. Navigation

Bottom navigation with 4 tabs: **Home · Library · Tuner · Settings**.
- A **mini player** sits above the tab bar on Home and Library while something is loaded: art, title, "artist · key", play/pause, dismiss, and a 2px progress line on top. Tapping it opens the Player.
- Pushed (no tab bar): Player, Piano learning view, YouTube search/detail, Processing, Recorder, Account migration.
- Sheet: Import confirmation.

> ⚠ **Open: "Tuner" tab.** The prototype routes it to the Recorder. An instrument tuner is a new feature that isn't in PLAN.md. See TASKS T-505. Until it's decided, apps ship **3 tabs** (Home, Library, Settings), with Record reachable from Home.

---

## 3. Screens

### 3.1 Home (`HomeScreen`)
- **Brand header**: logo mark (wave icon on a `primary` rounded square, rotated -4°), the "MusicDecoded" wordmark, tagline "Your music, decoded.", plus Search and Profile (→ Settings) buttons.
- **Recently Decoded**: up to 3 song rows plus "See all" → Library.
  - **Song row**: 56×56 art (radius 10) · title · artist · meta line `duration · key · BPM` · offline badge (download icon on `primarySoft`) when audio is local · ⋯ menu (Delete, Favorite, Re-analyze).
- **Transcribe Music**: a 2×2 grid of **source cards** (icon tile, title, description):
  Files ("Import audio from this device"), Media Library ("Choose local music"), YouTube ("Discover & play"), SoundCloud ("Find supported tracks"; hidden unless `soundcloudEnabled`).
- **Record card** (full width): mic icon, "Record Audio", "Capture a live performance", chevron.
- **Tip card** (sparkle icon, chord color): rotating tips, e.g. "Practice smarter — Slow down any decoded song without changing its key."
- Empty history: replace the Recently Decoded list with a one-line prompt; the source grid stays.

### 3.2 Library (`LibraryScreen`)
- Header: overline "YOUR COLLECTION" + title "Library" + search.
- Search field ("Songs, artists, keys..."), filtering local items by title, artist and key.
- Filter chips: **All songs · Favorites · Offline · Recordings** (selected = `primary` fill).
- Summary card: decoded count · available-offline count · grid/list toggle (grid view is post-v1; the toggle is hidden in v1).
- Sorted list ("Recently added") of song rows.

### 3.3 Import sheet (`ImportSheet`)
Shown after picking from Files or Media Library: file name, type · size, Duration / Quality / Storage, a note, **"Import & decode"** (primary) and **"Choose another file"** (secondary).
> Copy fix: the prototype says "analyze this file **on your device**". v1 analyzes on the server (PLAN D-1). Use: "MusicDecoded uploads this file to our servers to analyze it and create an interactive chord timeline. A copy stays on your device for offline playback." (The upload is kept so a signed-in user can sync their own audio to other devices; the privacy policy, T-504, must say so.)

### 3.4 YouTube (`YouTubeScreen`)
- **Search**: search field (emphasized border), suggestion chips (recent queries), "Top results" list. Result: 132×78 thumbnail with a duration badge · title (2 lines) · channel · views · age · ⋯.
- **"Paste a video URL"** card at the bottom → parses a videoId and goes to detail.
- **Video detail**: hero thumbnail (202 tall, play button plays the embed inline), title, channel · views, an **availability card** ("Analysis available" ✓), **"Decode this song"** primary button → Processing, and a **policy note**: "Offline audio is saved only when permitted. The chord timeline can remain available separately." (This copy *is* PLAN R-1. Keep it.)

### 3.5 Processing (`ProcessingScreen`)
- Art with two orbit rings plus a wave badge; title "Decoding your music" → "Your transcription is ready".
- Overall progress bar (from `Job.progress`).
- Five steps, mapped from `Job.stage`:

| UI step | `Job.stage` values |
|---|---|
| Preparing audio | `fetching_media`, `normalizing_audio` |
| Detecting beats & tempo | `detecting_beats` |
| Analyzing harmony | `detecting_chords`, `detecting_key` |
| Generating chord timeline | `rendering_spectrogram` |
| Saving transcription | `saving_results` |

- Buttons: "Continue in background" (→ Home; the row keeps showing progress) / "Open transcription" when complete. Close (×) = continue in background, which doesn't cancel the job.

### 3.6 Recorder (`RecorderScreen`)
Mic state dot ("Microphone ready" / red "Recording"), a large timer `00:MM:SS` (light weight, tabular numbers), live waveform (dimmed when idle), a 16-segment **input level** meter with a dB value, actions **Discard · Record/Pause (84pt red ring) · Finish ✓**, a "Save" text button in the header, and a tip about background noise. Finish/Save → title prompt → Processing.

### 3.7 Player (`PlayerScreen`): main screen
Top to bottom:
1. **Header**: back · title/artist (centered) · favorite (heart, toggles `favorite`) · ⋯.
2. **Metadata row**: three equal chips **Key** ("B minor") · **Time signature** ("4/4") · **BPM** ("140 BPM"), each with an icon. Keep layout space for future chips.
3. **Timeline** (`timelineBg` card):
   - time labels row;
   - **chord track**: rounded blocks spanning each chord segment, labelled with the display name. The current block is filled `chord`, the others `chordMuted`. **Tapping a block seeks to its start**;
   - **waveform**: vertical bars from `waveformPeaks` (played = primary). Spectrogram rendering is an optional alternate (`spectrogram` field) and isn't required in v1. Faint gridlines on downbeats;
   - **playhead**: a 1.5pt primary line with a dot on top and a soft glow;
   - hint "Pinch to zoom · drag to scrub". **Behavior**: content scrolls under a **fixed center playhead** while playing (PLAN §3.3), drag to scrub, pinch to zoom (60–400 pt/s, default 120).
4. **Current chord**: overline "NOW PLAYING", big chord name (34, chord color), note names ("B · D · F♯") from `chord-voicings.json`, and a caption "Representative root-position voicing".
5. **Control chips** (3 across, label above value + chevron): **Chords** (Basic / Full) · **Voicing** (Simple, v1 only) · **Transpose** (−6…+6). See §4.
6. **Visualizer card**: segmented **Piano | Video** plus an expand button (→ Piano learning view).
   - Piano: "CURRENT CHORD" + name + notes, a 2-octave keyboard (96 tall) with active keys highlighted.
   - Video: the YouTube embed (16:9) with a "Authorized embedded playback" badge. **Disabled unless source = youtube.**
7. **Transport** (bottom sheet, `transportBg`, radius 22 top): seek bar with current/total times; controls **speed** (1.00× ⇄ 0.75×; long-press for 0.5–1.25 in 0.05 steps) · **previous chord** · **play/pause** (58pt primary circle) · **next chord** · **loop**. When loop is on, a **loop pill** ("Looping 1:04–1:20 · Clear") floats above the transport.

### 3.8 Piano learning view (`PianoScreen`)
For practicing. Header with title/artist.
- **Head card**: CURRENT CHORD (name + notes) | UP NEXT (name + "in N beats", computed from `beats[]`).
- Position bar with times.
- **Large keyboard** (190 tall, edge to edge) with note names on the white keys (toggleable).
- **Chord progression strip**: horizontally scrolling cards (chord + duration in beats). The current one is filled `chord`, and it auto-scrolls.
- **Learning controls** (2×2): Speed · Loop · Note names · Transpose.
- Big transport (prev chord · play · next chord) and a hint "Rotate your device for a wider keyboard". Landscape shows a wider keyboard (more octaves).

### 3.9 Settings (`SettingsScreen`)
- Header: overline "PREFERENCES & ACCOUNT" + "Settings".
- **Profile card**: guest ("Guest account · Music is stored on this device" + "Create account") or account (name, email · Synced, ✓).
- Guest only: a **Sync card** ("Sync your music — Keep eligible chord timelines and files available across your devices.").
- Groups (rows: label · value · chevron):
  - **Playback**: Default speed · Autoplay next song · Remember loop regions
  - **Music analysis**: Chord notation (Standard) · Accidentals (Auto/♯/♭) · Default visualization (Piano/Video)
  - **Storage**: Local storage (size) · Offline files (count) · Clear cache
  - **Application**: Notifications · Privacy & data · Help & support (Appearance is hidden in v1 because the app is dark-only)
- v1 builds only the rows backed by behavior: Default speed, Chord notation, Accidentals, Default visualization, Local storage, Offline files, Privacy & data, Help. Hide the rest.

### 3.10 Account migration (`SettingsScreen` → migrate)
Shown after signup/login on a device with guest data ("Move local music"): a cloud icon, "Bring your music with you", and a checklist:
- ☑ **N chord timelines** (analysis, tempo and key data)
- ☑ **N local audio files** (eligible for private sync · size). These are file/recording sources.
- ☐ (disabled) **N restricted tracks** ("Timelines sync; source audio stays local"). These are youtube/soundcloud sources.
- Note: "Nothing is removed from this device. Restricted third-party audio never uploads."
- **Confirm & sync** → links the guest data and runs the first sync (PLAN §5.2).

### 3.11 Paywall (Pro, M6)
Not in the Figma prototype; built from the existing tokens and components so it matches the rest of the app.
- **Entry points**: the daily-limit error from `POST /jobs` (429 `rate_limited`, "You've used today's N transcriptions"), a "Go Pro" row in Settings (accounts only), and later any Pro-only feature (403 `pro_required`). **Never** from the YouTube search quota 429.
- **Layout** (sheet, dark): sparkle icon on `primarySoft`, title "MusicDecoded Pro", one-line subtitle "Decode more, practice deeper."; a comparison list of 3–4 rows from PLAN.md §9 (Free vs Pro: daily transcriptions "N / day" vs "Unlimited*", premium analysis models, sync of your own audio files); plan options from the store as selectable cards (monthly / yearly; yearly shows the saving), using the store's localized price strings, never hard-coded prices; **Subscribe** (PrimaryButton); "Restore purchases" (SecondaryButton/text); fine print: auto-renewal terms + links to Terms and Privacy (T-504 URLs; hidden until they exist) and "* fair-use limit applies".
- **Guests**: the Subscribe button reads "Create an account to subscribe" and opens sign-up first; the purchase continues after sign-up.
- **States**: loading offerings (skeleton cards), purchase in progress (button spinner, sheet not dismissible), success (checkmark, "You're Pro", dismiss → app re-fetches `/config`), cancelled (silent), error (inline PolicyNote-style message with Retry), purchases unavailable (no RevenueCat key in this build → "Purchases aren't available in this build").
- **Pro state elsewhere**: Settings profile card shows a "Pro" chip; the daily-limit message is never shown when `limits.dailyJobs` is null.

---

## 4. Player behavior (both apps)

- **Clock**: local audio position; if there's no local audio, the YouTube player's `currentTime`.
- **Current chord**: the segment where `start ≤ t < end`. **Up next**: the next segment whose label isn't `N`, with "in N beats" = number of `beats[].t` between t and its start.
- **Previous/next chord**: seek to the start of the previous/next non-`N` segment (previous: if more than 1s into the current one, go to its start first).
- **Speed**: time-stretch without changing pitch. iOS `AVPlayer.rate` with `audioTimePitchAlgorithm = .timeDomain`; Android `ExoPlayer.setPlaybackParameters(PlaybackParameters(speed, 1f))`. YouTube video uses `setPlaybackRate`.
- **Loop**: tapping loop when it's off loops the **current bar** (between downbeats). Dragging on the chord track sets a custom range. The range snaps to beats.
- **Transpose** (−6…+6 semitones) shifts chord labels, note names and piano keys. **Display only in v1**, not audio pitch.
- **Chords: Basic** simplifies each label to a triad family: maj/7/maj7/9… → major, min/min7/min9… → minor, keeping dim/aug/sus. **Full** shows the label as analyzed.
- **Accidentals**: Auto uses the key signature (flat keys → ♭), otherwise forced ♯ or ♭.
- **Chord display names**: `A:maj`→`A`, `F#:min7`→`F♯m7`, `D:maj/3`→`D/F♯`, `hdim7`→`m7♭5`, `dim`→`°`, `aug`→`+`, `N`→none. Use real ♯/♭ glyphs in the UI.
- **Note names** under the chord come from `contracts/chord-voicings.json`. The caption "Representative root-position voicing" makes clear these aren't the exact notes played.

---

## 5. Out of scope in v1 (present in the prototype)
Tuner (pending decision), Library grid view, Appearance/light theme, Notifications, Autoplay next song, "Voicing" options beyond Simple, global search on Home (the header search button opens Library search instead).
