package com.musicdecoded.mobile.core

private val MAJOR_NUMERALS = listOf("I", "♭II", "II", "♭III", "III", "IV", "♯IV", "V", "♭VI", "VI", "♭VII", "VII")
private val MINOR_NUMERALS = listOf("I", "♭II", "II", "III", "♯III", "IV", "♯IV", "V", "VI", "♯VI", "VII", "♯VII")

/**
 * The chord's Roman numeral relative to [key] (DESIGN.md §4 "Roman numerals"), e.g. key C major:
 * `D:min7` → `ii`, `B:hdim7` → `viiø`, `Bb:maj` → `♭VII`. Uses the chord root (not the slash bass).
 * `null` for `N`/`X`, unparsable labels or no key. Transpose shifts key and chord equally, so it
 * isn't a parameter.
 */
fun romanNumeral(label: String, key: Analysis.Key?): String? {
    if (key == null) return null
    val chord = ChordLabel.parse(label) ?: return null
    val root = PitchSpelling.pitchClass(chord.root) ?: return null
    val tonic = PitchSpelling.pitchClass(key.tonic) ?: return null
    val degree = wrapPitchClass(root - tonic)
    val numeral = (if (key.mode == Analysis.Mode.MINOR) MINOR_NUMERALS else MAJOR_NUMERALS)[degree]
    val quality = chord.quality.substringBefore('(')
    return when {
        quality.startsWith("hdim") -> numeral.lowercase() + "ø"
        quality.startsWith("dim") -> numeral.lowercase() + "°"
        quality.startsWith("min") -> numeral.lowercase()
        quality.startsWith("aug") -> "$numeral+"
        else -> numeral
    }
}
