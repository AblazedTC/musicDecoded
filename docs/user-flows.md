# User Flows

## Core Flow

```
1.  User logs in
2.  User pastes a YouTube link
3.  Android app sends POST /tracks/youtube to Spring Boot
4.  Spring Boot extracts the YouTube video ID
5.  Spring Boot checks if this track already exists
6.  If track exists → reuse it
    If track does not exist → create it in PostgreSQL
7.  Add the track to the current user's library
8.  Check if analysis already exists
9.  If analysis exists → return completed status
    If analysis does not exist → create an analysis job
10. Push job message to queue
11. Return jobId to Android app
12. Android app shows processing status screen
13. Python worker consumes the job
14. Worker calls youtube-video-downloader-api
15. Worker downloads temporary audio/video
16. Worker extracts and normalizes audio using FFmpeg
17. Worker runs music analysis models
18. Worker saves JSON artifacts to object storage
19. Worker updates job status to complete
20. Worker deletes temporary media files
21. Android app fetches completed analysis
22. Android app plays YouTube video and syncs UI to timestamps
```

---

## Playback / Analysis Separation

The project separates **playback** from **analysis**. These are two different sources:

|          | Source                                                       |
| -------- | ------------------------------------------------------------ |
| Playback | YouTube player (embedded in app)                             |
| Analysis | Temporary downloaded media from youtube-video-downloader-api |

The app does not store or stream audio itself. It replays the YouTube video and syncs the analysis UI to the player's current timestamp.

---

## Android App Screens

```
Login screen
Signup screen
Library screen
Submit YouTube Link screen
Processing screen (polls job status)
Track Detail screen
  └── Waveform + Chords view
  └── Beat Grid view
  └── Theory Summary view
```

---

## Playback Sync

The YouTube player is the source of truth for the current time.

On every tick, the app reads `currentTime` from the player, then looks up:

- Current chord
- Current beat
- Current bar
- Current lyric line (if available)

```kotlin
fun getCurrentMusicState(
    currentTime: Double,
    analysis: TrackAnalysis
): CurrentMusicState

data class CurrentMusicState(
    val currentChord: String?,
    val currentBar: Int?,
    val currentBeat: Int?,
    val currentLyricLine: String?
)
```

The UI uses this state to highlight the active chord, grid cell, and waveform position.

---

## Analysis Views

### 1. Waveform / Chord Timeline

The casual playback view. Shows the song moving over time.

```
[YouTube Player]
[Waveform timeline]
|----F#m7----|----Dmaj7----|----A----|----E----|
                    ^
               current time
```

### 2. Beat Grid

The musician/theory view. For 4/4 time:

```
Bar 1: [1] [2] [3] [4]   F#m7
Bar 2: [1] [2] [3] [4]   Dmaj7
Bar 3: [1] [2] [3] [4]   A
Bar 4: [1] [2] [3] [4]   E
```

The active square highlights based on YouTube playback time.

---

## Python Worker Stages

```
queued
resolving_youtube_media
downloading_temp_media
extracting_audio
normalizing_audio
generating_waveform
detecting_bpm
detecting_beats
detecting_downbeats
detecting_key
detecting_chords
building_grid
generating_explanation
saving_results
complete
failed
```

### Internal audio format

All input media is normalized to WAV before model inference, regardless of what the downloader returns:

```
Input (any): mp3 / m4a / webm / opus / mp4
                    |
                  FFmpeg
                    |
Output:    normalized.wav (mono, 22050 Hz, 16-bit PCM)
```

This means the ML pipeline never needs to handle different input formats.
