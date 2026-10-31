package com.musicdecoded.mobile.player

import androidx.compose.runtime.Immutable
import com.musicdecoded.mobile.core.Accidentals
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.ChordVoicings
import com.musicdecoded.mobile.core.PianoVoicing
import com.musicdecoded.mobile.core.PitchSpelling
import com.musicdecoded.mobile.core.chordDisplayName
import com.musicdecoded.mobile.core.romanNumeral

/**
 * How chords are shown with the current Chords (Basic/Full) and Transpose settings, so the
 * timeline, Now Playing and the piano always agree (DESIGN.md §4).
 */
@Immutable
data class ChordDisplay(
    val key: Analysis.Key?,
    val transpose: Int,
    val basic: Boolean,
    val voicings: ChordVoicings?,
    /** Settings → Accidentals. */
    val accidentals: Accidentals = Accidentals.AUTO,
) {
    private val spelling = PitchSpelling(accidentals, key, transpose)

    /** `F#:min7` → `F♯m7` (Full) or `F♯m` (Basic); `""` for `N`/`X`. */
    fun name(label: String): String = chordDisplayName(label, transpose, accidentals, key, simplify = basic)

    /** Roman numeral in the detected key (`III`, `viiø`); `null` for `N`/`X` or no key. Unaffected by transpose. */
    fun numeral(label: String): String? = romanNumeral(label, key)

    /**
     * The fixed-keyboard voicing (DESIGN.md §4 "Piano voicing & inversions") at [inversion], which
     * is clamped to the chord's number of inversions; `null` for `N`/`X`.
     */
    fun voicing(label: String, inversion: Int = 0): PianoVoicing? {
        val voicings = voicings ?: return null
        val root = PianoVoicing.of(voicings, label, 0, transpose, basic) ?: return null
        return if (inversion <= 0) root else PianoVoicing.of(voicings, label, inversion.coerceAtMost(root.inversions - 1), transpose, basic)
    }

    /** `C`, `F♯`, `B♭`… spelled for the (transposed) key. */
    fun noteName(midi: Int): String = spelling.name(midi)

    /** `E · G♯ · B` (each pitch class once, bass first). */
    fun noteNames(label: String, inversion: Int = 0): String =
        voicing(label, inversion)?.let { listOf(it.bass) + it.rightHand }.orEmpty().map(::noteName).distinct().joinToString(" · ")
}

/** Transpose chip value: `0`, `+2`, `−3` (real minus sign). */
fun formatTranspose(semitones: Int): String = when {
    semitones > 0 -> "+$semitones"
    semitones < 0 -> "−${-semitones}"
    else -> "0"
}

/** `1.00×`. */
fun formatSpeed(speed: Float): String = "%.2f×".format(java.util.Locale.ROOT, speed)
