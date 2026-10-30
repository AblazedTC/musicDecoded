# DESIGN.md — Screens, tokens & player spec

> **Visual style: "Wrapped"**, taken from the owner's portfolio (https://jordanjerkins.dev, repo `SpotiPortfolioJordan/spotiwrap`: `tailwind.config.ts`, `src/styles/globals.css`, `src/components/story/*`, esp. `Sticker.tsx` and `NowPlayingCard.tsx`). Neo-brutalist Spotify-Wrapped look: flat saturated color blocks, thick black borders, hard offset shadows, tilted stickers, huge uppercase display type.
> **Layout and screen content** still come from the Figma Make prototype (fileKey `EEN7dqUVB7LlQFTNiFKvHz`, section 3 below). Where the two disagree on *looks*, section 1 wins; on *what is on a screen*, section 3 wins.

Both apps implement the same screens and behavior. Platform conventions (navigation, sheets, back gestures, haptics) follow each OS.

---

## 1. Design tokens

**Dark-only in v1**, but "dark" here means **ink surfaces with loud color blocks**, not a navy UI. Every pairing below was checked for WCAG AA (4.5:1 for text); keep to the contrast rules.

### Colors
Palette (exact portfolio values):

| Token | Hex | Role |
|---|---|---|
| `violet` | `#4100F5` | hero/brand blocks (Home header, paywall, Processing) |
| `lime` | `#CFF56A` | main accent: playhead, progress, current chord, selected tab, eyebrow text on dark |
| `pink` | `#F774C4` | second accent: play button, loop pill, now-playing highlights, piano bass note |
| `yellow` | `#FAE62D` | tags/badges ("PRO"), offline badge, warnings |
| `orange` | `#FF6437` | record button, destructive actions, errors |
| `blue` | `#509BF5` | links, info states, Media Library source |
| `ink` | `#121212` | default screen background; text on color blocks |
| `card` | `#1C1C1C` | cards/rows on ink |
| `cardRaised` | `#262626` | pressed state, inputs, progress track |
| `white` | `#FFFFFF` | text on ink/violet; white cards on violet |
| `cream` | `#F0EDE4` | piano white keys, light paper cards |

Semantic tokens (apps keep these names; values change):

| Token | Value | Notes |
|---|---|---|
| `canvas` | `ink` | screen background |
| `surface` / `surfaceRaised` | `card` / `cardRaised` | rows, sheets, chips on dark |
| `primary` / `onPrimary` | `lime` / `ink` | |
| `secondary` / `onSecondary` | `pink` / `ink` | |
| `chord` / `onChord` | `lime` / `ink` | current chord block, big chord name on dark |
| `chordMuted` | bg `#2A2A2A`, border `#3A3A3A` | inactive chord blocks |
| `text` | `white` on ink/violet; `ink` on lime/pink/yellow/orange/blue/cream | |
| `textSecondary` | `#B3B3B3` on ink (8.9:1), `#D6CCFF` on violet (5.5:1), `#3A3A3A` on lime (9.2:1) | |
| `textMuted` | `#8A8A8A`, on ink only (5.4:1) | captions, inactive tabs |
| `border` | black 3pt on color blocks; white at 20% (3pt) around art/thumbnails on ink | |
| `record` | `orange` | |
| `waveIdle` / `wavePlayed` | `#3A3A3A` / `lime` | waveform bars |

Contrast (computed): white on violet 8.3, lime on violet 6.7, yellow on violet 6.5, ink on lime 15.1, ink on pink 7.4, ink on yellow 14.7, ink on orange 6.4, ink on blue 6.6; lime/pink/yellow/blue/orange on ink all at least 6.4. **Never** pink text on violet (3.2): pink on violet only as a filled shape with ink text.

Source tiles (fill / icon): Files `lime`/ink, Media Library `blue`/ink, YouTube `orange`/ink, SoundCloud `yellow`/ink, Record `pink`/ink.

Piano: white keys `cream` with ink 1.5pt separators, black keys `ink`, active white key `lime`, active black key `#9DBF2E`, bass note `pink`, keyboard frame 3pt black border + hard shadow.

### Shape, borders & elevation
- **Corners are square.** Radius 0 for cards, rows, buttons, chips, sheets and inputs. Only circular play/record buttons and avatars are round.
- **Border**: 3pt black on any color block or card on a color block; 3pt white at 20% around thumbnails/art on ink. Separators inside lists: 2pt.
- **Hard shadow** (no blur, offset x = y): `shadowSm` 3pt (chips, small buttons), `shadowMd` 5pt (stickers, buttons, rows on color), `shadowLg` 8pt (cards, timeline, visualizer), `shadowXl` 10pt (hero cards: player now-playing card, paywall). Black on color blocks; on ink use an **accent-colored** shadow (`violet` or `pink`) so it stays visible.
- **Tilt**: hero cards -1.5 deg, list cards alternate -1/+1 deg, stickers -3 deg (default) to -6 deg. Decoration only: never tilt the timeline, piano, inputs or anything draggable. No tilt when Reduce Motion is on.
- **Pressed state**: move by (+2, +2) and shrink the hard shadow by 2pt (the "pushed sticker" feel); no ripple tint on color blocks.

### Signature elements (shared components)
- **Sticker** (portfolio `Sticker.tsx`): accent fill, 3pt black border, `shadowMd`, uppercase heavy body font, +0.04em tracking, tilted -3 deg. For section eyebrows ("YOUR LIBRARY", "NOW PLAYING"), status badges, "PRO".
- **Eyebrow label**: uppercase, at least 11pt, bold, tracking +0.25em; `lime` on ink, `white` on violet.
- **Display headline**: Archivo Black, uppercase, line height 0.95, tracking -0.01em.
- **Big numerals**: oversized numbers at 8% opacity ("01", "#1") behind section titles and Library rows; decorative, hidden from accessibility.
- **Waveform bars**: simple vertical bars (portfolio `Waveform.tsx`) for idle/now-playing indicators.
- **Textures** (at most one per screen): halftone dot grid (2pt dots, 12pt pitch, 20% opacity), diagonal stripes (violet at 50% on lime blocks).

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
Bundle two Google Fonts (SIL Open Font License, free to ship in apps): **Archivo Black** (display) and **Space Grotesk** (body; weights 400/500/700). Keep **Dynamic Type / font scaling**: `Font.custom(_:size:relativeTo:)` on iOS, sp-based `TextUnit` with a `FontFamily` on Android. Minimum size 11pt/sp.

| Role | Font | Default size | Notes |
|---|---|---|---|
| Page title | Archivo Black | 44 | uppercase, line height 0.95 |
| Section title | Archivo Black | 28 | uppercase |
| Current chord (player) | Archivo Black | 56 | lime on ink |
| Card / row title | Space Grotesk Bold | 17 | song titles in sentence case |
| Body | Space Grotesk Regular | 15 | |
| Secondary | Space Grotesk Medium | 13 | `textSecondary` |
| Eyebrow / caption | Space Grotesk Bold | 11 | uppercase, tracking +0.25em |
| Times / counters / BPM | Space Grotesk Medium | 13 | tabular figures |

### Icons
Bold line icons (2pt stroke): closest SF Symbol / Material Symbols **Sharp** (square ends match the brutalist look). Icons on color blocks are `ink`. YouTube/SoundCloud marks follow their brand rules.

### Screen color map
| Screen | Background | Accent |
|---|---|---|
| Home | violet hero block (title, stickers, waveform), then ink body with source tiles in their fills | lime |
| Library | lime header block (ink text, stripe texture), ink list | violet |
| Processing | violet; step list on an ink card with `shadowXl` | lime progress |
| Player | ink; now-playing card (`card`, 3pt black border, pink `shadowXl`, tilted -1.5 deg) | lime chords/playhead, pink play |
| Piano learning view | ink; keyboard frame with lime `shadowLg` | lime |
| Recorder | ink; orange record button with black border + `shadowMd` | orange |
| Settings / Auth / Migration | ink; profile card as a yellow block | lime |
| Paywall | violet; white plan cards (black border + `shadowLg`); yellow "PRO" sticker | lime CTA |
| Tab bar | ink with 3pt black top border; selected = lime icon + label, unselected `textMuted` | |

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
  - **Song row**: 56×56 art (square, white-20% 3pt border) · title · artist · meta line `duration · key · BPM` · offline badge (yellow sticker with download icon) when audio is local · ⋯ menu (Delete, Favorite, Re-analyze).
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
**Owner revision (T-214, 2026-10-30).** Layout reference: a single-screen chord player (compact header, edge-to-edge waveform with chord-name + Roman-numeral boxes, compact control row, piano filling the rest, transport at the bottom). Keep the §1 Wrapped colors, type, icons and borders; this section only changes **placement, sizing and hierarchy**.

**The whole player fits on one screen with no vertical scrolling** (target: 640pt/dp tall viewports and up, after safe areas). Top to bottom:

1. **Header** (one row, ~48): back · title (one line, ellipsized; artist under it only if it fits, else omitted) · favorite · ⋯. No extra vertical padding.
2. **Metadata row** (one row, ~36): Key · Time · BPM as compact chips with the value inline next to a small icon (no separate "KEY"/"TIME"/"TEMPO" label line). Room for future chips.
3. **Timeline**, **full width, edge to edge**: no card, no outer horizontal margin, no offset shadow, no border on the left/right. Fixed height ~32% of the space between the metadata row and the transport (clamped 150-220). Inside, top to bottom:
   - a thin time-label row (small `textMuted` ticks every second at default zoom);
   - **chord labels** above the waveform (see below);
   - the **waveform** (existing styling: played `lime`, unplayed `waveIdle`, downbeat gridlines);
   - the **playhead**: fixed at the horizontal center, 2pt `lime` line through labels and waveform.
   - Pinch to zoom (60-400 pt/s, default 120) and drag to scrub still work; **no hint text row** (the hint becomes the timeline's accessibility hint).
   - **Chord labels**: one compact box per chord change (not spanning the chord's duration), its left edge anchored at the change time, a 1pt tick down to the waveform. Two text lines, centered: chord display name (Archivo Black ~20) and its **Roman numeral** relative to the detected key (Space Grotesk Bold ~15; §4 "Roman numerals"). Box: square, 2pt black border, padding 4/6; **current chord** = `lime` fill + `ink` text + `shadowSm`; others = `cardRaised` fill + `white` text. `N`/`X` segments get no box. No key detected → name only.
   - **Collision handling**: labels are laid out in time order into up to **2 staggered rows**; a label goes in the first row where it doesn't overlap the previous label in that row (4pt gap). If neither row fits, the label is dropped at the current zoom (its tick stays); the current chord's label is never dropped (it replaces the overlapping neighbor). Re-run on zoom.
   - Labels, ticks, waveform and playhead stay in sync during playback, seeking and zooming (same time→x mapping). Tapping a label seeks to that chord.
4. **Control row** (one row, ~44, 8 gap): **Chords** (Basic/Full) · **Voicing** (Simple) · **Transpose** (-6...+6). Same styling as before with tighter padding (label + value on one line where width allows).
5. **Visualizer panel** takes **all remaining height**:
   - compact tab header (~36): **Piano | Video** segmented (Piano selected by default; Video disabled unless source = youtube) + expand button (to the Piano learning view);
   - **Piano**: the **active chord name** centered above the keyboard (Archivo Black ~28, `lime`; Roman numeral next to it in `textSecondary`), then the keyboard sized to fill the rest of the panel (min 96, keys keep proportions; show as many octaves as fit the width, at least 2, centered on the active notes). Highlighted keys show the notes; note names (e.g. "E · G♯ · B") under the highlighted keys in small text. This replaces the old Now Playing card.
   - **Video**: 16:9 embed letterboxed in the available space.
6. **Transport**, anchored at the bottom above the bottom safe area (compact, ~104 total): seek bar with current/total times on one line, then speed · previous chord · play/pause (56 pink circle) · next chord · loop. The loop pill floats above the transport without covering the visualizer's keys (overlap the panel header area if needed).

**Removed**: the "Now Playing" card (chord name, notes, "Representative root-position voicing" caption) and the timeline's inset card/shadow/hint row.
**Accessibility**: at the largest text sizes the screen may scroll as a fallback; the timeline and keyboard keep their own accessibility summaries ("Current chord E major, three. Next A major, six.").

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

### Roman numerals (timeline labels, piano panel)
Computed from the analysis `key` and each chord label (never hard-coded); identical on both apps (shared test cases).
- **Degree** = semitones from the key tonic to the chord root (after transpose both shift equally, so numerals don't change with transpose).
- **Major key** numerals by interval: 0 `I`, 1 `♭II`, 2 `II`, 3 `♭III`, 4 `III`, 5 `IV`, 6 `♯IV`, 7 `V`, 8 `♭VI`, 9 `VI`, 10 `♭VII`, 11 `VII`.
- **Minor key** (relative to the natural minor scale): 0 `I`, 1 `♭II`, 2 `II`, 3 `III`, 4 `♯III`, 5 `IV`, 6 `♯IV`, 7 `V`, 8 `VI`, 9 `♯VI`, 10 `VII`, 11 `♯VII`.
- **Case = chord quality**: uppercase for major/dominant/augmented/sus/power chords (`maj`, `7`, `maj7`, `9`, `aug`, `sus2`, `sus4`, `5`, ...); lowercase for minor (`min`, `min7`, `min9`, ...) and diminished; append `°` for `dim`/`dim7`, `ø` for `hdim7`, `+` for `aug`. No extensions (7, 9) in the numeral.
- Slash chords use the chord root, not the bass. `N`/`X` and missing key → no numeral.
- Examples: key C♯ minor: `E:maj` → `III`, `A:maj` → `VI`, `C#:min` → `i`, `G#:7` → `V`, `B:maj` → `VII`. Key C major: `D:min7` → `ii`, `G:7` → `V`, `B:hdim7` → `viiø`, `Bb:maj` → `♭VII`, `F:min` → `iv`.

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
