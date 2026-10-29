package com.musicdecoded.mobile.core

/** How black-key notes are spelled in chord names (DESIGN.md §4, Settings → Accidentals). */
enum class Accidentals {
    /**
     * Follow the key signature: flat keys use ♭, sharp keys use ♯. With no key, or in
     * C major / A minor, use the common chart spelling (C♯ E♭ F♯ A♭ B♭).
     */
    AUTO,
    SHARP,
    FLAT,
}

/** Display suffix per Harte quality (DESIGN.md §4). `maj` has no suffix. */
private val qualitySuffixes = mapOf(
    "maj" to "", "min" to "m", "dim" to "°", "aug" to "+",
    "sus2" to "sus2", "sus4" to "sus4", "sus4(b7)" to "7sus4", "5" to "5",
    "7" to "7", "maj7" to "maj7", "min7" to "m7", "minmaj7" to "m(maj7)",
    "dim7" to "°7", "hdim7" to "m7♭5", "maj6" to "6", "min6" to "m6",
    "9" to "9", "maj9" to "maj9", "min9" to "m9",
    "11" to "11", "min11" to "m11",
    "13" to "13", "maj13" to "maj13", "min13" to "m13",
)

/**
 * The name shown in the player, e.g. `A:maj` → `A`, `F#:min7` → `F♯m7`, `D:maj/3` → `D/F♯`.
 * `N`/`X` → `""`.
 *
 * @param transpose semitones (−6…+6 in the UI); wraps around the octave.
 * @param accidentals ♯/♭ preference. [Accidentals.AUTO] reads [key] (shifted by [transpose]).
 * @param simplify Chords: Basic. Reduces the label to its triad family (major, minor, °, +,
 *   sus2/sus4, 5) and drops the slash bass.
 */
fun chordDisplayName(
    label: String,
    transpose: Int = 0,
    accidentals: Accidentals = Accidentals.AUTO,
    key: Analysis.Key? = null,
    simplify: Boolean = false,
): String {
    val chord = ChordLabel.parse(label) ?: return ""
    val rootPitch = PitchSpelling.pitchClass(chord.root) ?: return ""
    val spelling = PitchSpelling(accidentals, key, transpose)
    val root = rootPitch + transpose
    val quality = if (simplify) triadFamily(chord.quality) else chord.quality
    val name = StringBuilder(spelling.name(root) + suffix(quality))
    if (!simplify) {
        val degree = chord.bass?.let { PitchSpelling.semitonesOfDegree(it) }
        if (degree != null) name.append('/').append(spelling.name(root + degree))
    }
    return name.toString()
}

private fun suffix(quality: String): String =
    qualitySuffixes[quality]
        ?: qualitySuffixes[quality.substringBefore('(')]
        ?: quality.replace("b", "♭").replace("#", "♯")

/** Chords: Basic. maj/7/maj7/9… → maj, min/min7/min9… → min; keeps dim, aug, sus and 5. */
private fun triadFamily(quality: String): String = when {
    quality.startsWith("dim") || quality.startsWith("hdim") -> "dim"
    quality.startsWith("min") -> "min"
    quality.startsWith("aug") -> "aug"
    quality.startsWith("sus2") -> "sus2"
    quality.startsWith("sus4") -> "sus4"
    quality == "5" -> "5"
    else -> "maj"
}

/** Turns pitch classes into note names with real ♯/♭ glyphs. */
internal class PitchSpelling(accidentals: Accidentals, key: Analysis.Key?, transpose: Int) {
    private val names: List<String> = when (accidentals) {
        Accidentals.SHARP -> SHARP_NAMES
        Accidentals.FLAT -> FLAT_NAMES
        Accidentals.AUTO -> {
            val tonic = key?.let { pitchClass(it.tonic) }
            if (key == null || tonic == null) {
                COMMON_NAMES
            } else {
                // Compare keys through their relative major (A minor ↔ C major); transpose moves the key too.
                val relativeMajor = wrapPitchClass(tonic + transpose + if (key.mode == Analysis.Mode.MINOR) 3 else 0)
                when (relativeMajor) {
                    0 -> COMMON_NAMES
                    in FLAT_MAJOR_TONICS -> FLAT_NAMES
                    else -> SHARP_NAMES
                }
            }
        }
    }

    fun name(pitchClass: Int): String = names[wrapPitchClass(pitchClass)]

    companion object {
        private val SHARP_NAMES = listOf("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
        private val FLAT_NAMES = listOf("C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B")
        private val COMMON_NAMES = listOf("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")

        /** Major keys (by tonic pitch class) whose signature uses flats: F B♭ E♭ A♭ D♭. */
        private val FLAT_MAJOR_TONICS = setOf(5, 10, 3, 8, 1)
        private val NATURALS = mapOf('C' to 0, 'D' to 2, 'E' to 4, 'F' to 5, 'G' to 7, 'A' to 9, 'B' to 11)
        private val MAJOR_SCALE = listOf(0, 2, 4, 5, 7, 9, 11)

        /** Pitch class of a note name such as `C`, `F#`, `Bb` (ASCII accidentals, as in labels). */
        fun pitchClass(note: String): Int? {
            var value = NATURALS[note.firstOrNull() ?: return null] ?: return null
            for (accidental in note.drop(1)) {
                value += when (accidental) {
                    '#' -> 1
                    'b' -> -1
                    else -> return null
                }
            }
            return wrapPitchClass(value)
        }

        /** Semitones above the root for a Harte bass degree such as `3`, `b3`, `#4`, `b9`. */
        fun semitonesOfDegree(degree: String): Int? {
            val digits = degree.trimStart('b', '#')
            val accidentals = degree.take(degree.length - digits.length)
            val shift = accidentals.count { it == '#' } - accidentals.count { it == 'b' }
            val number = digits.toIntOrNull()?.takeIf { it >= 1 } ?: return null
            return MAJOR_SCALE[(number - 1) % 7] + 12 * ((number - 1) / 7) + shift
        }
    }
}
