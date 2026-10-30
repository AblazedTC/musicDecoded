package com.musicdecoded.mobile.ui.theme

import androidx.compose.ui.graphics.Color

/** DESIGN.md §1 color tokens. The app is dark-only in v1, so there is a single palette. */
object MusicDecodedColors {
    val Ink = Color(0xFF07101C)
    val Canvas = Color(0xFF0B1422)
    val Surface = Color(0xFF111F30)
    val SurfaceRaised = Color(0xFF182A3E)
    val SurfaceSoft = Color(0xFF20354B)
    val Line = Color(0xFF2A4057)

    val Primary = Color(0xFF62DCC4)
    val OnPrimary = Color(0xFF08221E)
    val PrimarySoft = Color(0xFF173D3D)
    val Secondary = Color(0xFF91AFFF)

    val Chord = Color(0xFFF2DFAF)
    val OnChord = Color(0xFF322B1D)
    val ChordMuted = Color(0xFF463F32)
    val ChordMutedBorder = Color(0xFF645943)

    val Text = Color(0xFFF5F8FB)
    val TextSecondary = Color(0xFF9FB0C3)
    /**
     * DESIGN.md says `#63778C`, which is only 3.6–4.1:1 on canvas/surface (WCAG AA needs 4.5:1 for
     * text, T-503a). Lifted to `#8396AA` (5.5–6.3:1 on canvas, surface, transport); the DESIGN.md
     * token change is logged as T-506. See ContrastTest.
     */
    val TextMuted = Color(0xFF8396AA)

    val Record = Color(0xFFEC6478)
    val RecordRing = Color(0xFFB94D5E)

    val TransportBg = Color(0xFF152638)
    val TransportBorder = Color(0xFF30465D)
    val MiniPlayerBg = Color(0xFF1B2D40)
    val TimelineBg = Color(0xFF09101A)
    val TimelineBorder = Color(0xFF24384B)
    val WaveIdle = Color(0xFF4E6379).copy(alpha = 0.72f)
    val WavePlayed = Primary.copy(alpha = 0.80f)

    // Source-icon tints (icon / background).
    val MediaLibraryBg = Color(0xFF1D3153)
    val YouTube = Color(0xFFFF8B8B)
    val YouTubeBg = Color(0xFF452534)
    val SoundCloud = Color(0xFFFFB56F)
    val SoundCloudBg = Color(0xFF432F26)
    val RecordSource = Color(0xFFFF9A9A)
    val RecordSourceBg = Color(0xFF412534)

    // Piano.
    val PianoWhite = Color(0xFFE7EBEC)
    val PianoWhiteBorder = Color(0xFF8D999D)
    val PianoBlack = Color(0xFF0C1218)
    val PianoWhiteActive = Color(0xFF9DE8DB)
    val PianoBlackActive = Color(0xFF48BEA8)
    val PianoBg = Color(0xFF0A111A)
}
