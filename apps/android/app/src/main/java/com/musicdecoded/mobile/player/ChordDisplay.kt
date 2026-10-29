package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Accidentals
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.ChordLabel
import com.musicdecoded.mobile.core.ChordVoicings
import com.musicdecoded.mobile.core.PitchSpelling
import com.musicdecoded.mobile.core.chordDisplayName
import com.musicdecoded.mobile.core.triadFamily

/**
 * How chords are shown with the current Chords (Basic/Full) and Transpose settings, so the
 * timeline, Now Playing and the piano always agree (DESIGN.md §4).
 */
data class ChordDisplay(
    val key: Analysis.Key?,
    val transpose: Int,
    val basic: Boolean,
    val voicings: ChordVoicings?,
) {
    private val spelling = PitchSpelling(Accidentals.AUTO, key, transpose)

    /** `F#:min7` → `F♯m7` (Full) or `F♯m` (Basic); `""` for `N`/`X`. */
    fun name(label: String): String = chordDisplayName(label, transpose, Accidentals.AUTO, key, simplify = basic)

    /** MIDI notes from `chord-voicings.json`; Basic voices the triad family without the slash bass. */
    fun notes(label: String): List<Int> = voicings?.notes(voicedLabel(label), transpose).orEmpty()

    /** The slash-chord bass (always the lowest voiced note), or `null`. */
    fun bassNote(label: String): Int? {
        val chord = ChordLabel.parse(voicedLabel(label)) ?: return null
        return if (chord.bass != null) notes(label).firstOrNull() else null
    }

    /** `C`, `F♯`, `B♭`… spelled for the (transposed) key. */
    fun noteName(midi: Int): String = spelling.name(midi)

    /** `B · D · F♯` (each pitch class once, lowest first). */
    fun noteNames(label: String): String = notes(label).map(::noteName).distinct().joinToString(" · ")

    private fun voicedLabel(label: String): String {
        if (!basic) return label
        val chord = ChordLabel.parse(label) ?: return label
        return "${chord.root}:${triadFamily(chord.quality)}"
    }
}

/** Transpose chip value: `0`, `+2`, `−3` (real minus sign). */
fun formatTranspose(semitones: Int): String = when {
    semitones > 0 -> "+$semitones"
    semitones < 0 -> "−${-semitones}"
    else -> "0"
}

/** `1.00×`. */
fun formatSpeed(speed: Float): String = "%.2f×".format(java.util.Locale.ROOT, speed)
