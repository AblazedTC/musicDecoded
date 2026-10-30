package com.musicdecoded.mobile.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * DESIGN.md §1 "Wrapped" color tokens: ink surfaces with loud color blocks. Dark-only in v1, so
 * there is a single palette. Screens use the semantic names; the raw palette is for color blocks.
 */
object MusicDecodedColors {
    // Palette (exact portfolio values).
    val Violet = Color(0xFF4100F5)
    val Lime = Color(0xFFCFF56A)
    val Pink = Color(0xFFF774C4)
    val Yellow = Color(0xFFFAE62D)
    val Orange = Color(0xFFFF6437)
    val Blue = Color(0xFF509BF5)
    val Ink = Color(0xFF121212)
    val Card = Color(0xFF1C1C1C)
    val CardRaised = Color(0xFF262626)
    val White = Color(0xFFFFFFFF)
    val Cream = Color(0xFFF0EDE4)
    val Black = Color(0xFF000000)

    // Semantic tokens.
    val Canvas = Ink
    val Surface = Card
    val SurfaceRaised = CardRaised

    val Primary = Lime
    val OnPrimary = Ink
    val Secondary = Pink
    val OnSecondary = Ink

    val Chord = Lime
    val OnChord = Ink
    val ChordMuted = Color(0xFF2A2A2A)
    val ChordMutedBorder = Color(0xFF3A3A3A)

    /** Text on ink and violet. On lime/pink/yellow/orange/blue/cream use [OnColor]. */
    val Text = White
    /** Text on every light color block (lime, pink, yellow, orange, blue, cream). */
    val OnColor = Ink
    val TextSecondary = Color(0xFFB3B3B3)
    val TextSecondaryOnViolet = Color(0xFFD6CCFF)
    val TextSecondaryOnLime = Color(0xFF3A3A3A)
    /** Captions and inactive tabs, on ink only. */
    val TextMuted = Color(0xFF8A8A8A)

    /** 3pt border on color blocks and cards on color blocks. */
    val Border = Black
    /** 3pt border around thumbnails/art on ink: white at 20%. */
    val BorderOnInk = White.copy(alpha = 0.20f)
    /** Hairline separators and quiet outlines on ink (same as the inactive chord border). */
    val Line = ChordMutedBorder

    val Record = Orange
    val Info = Blue
    val Warning = Yellow

    val WaveIdle = Color(0xFF3A3A3A)
    val WavePlayed = Lime

    // Piano.
    val PianoWhite = Cream
    val PianoBlack = Ink
    val PianoWhiteActive = Lime
    val PianoBlackActive = Color(0xFF9DBF2E)
    // Bass (left hand) = blue per DESIGN.md §3.7 owner revision 2.
    val PianoBass = Blue
}
