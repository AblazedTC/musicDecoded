package com.musicdecoded.mobile.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class ChordDisplayNameTest {
    /** DESIGN.md §4 display table, plus every quality the fixture uses. */
    @Test fun fullNames() {
        val cases = listOf(
            "A:maj" to "A", "F#:min7" to "F♯m7", "D:maj/3" to "D/F♯", "B:hdim7" to "Bm7♭5",
            "B:dim" to "B°", "C:aug" to "C+", "N" to "", "X" to "", "G:7" to "G7",
            "C:maj7" to "Cmaj7", "E:min" to "Em", "A:min9" to "Am9", "A:sus4(b7)" to "A7sus4",
            "E:min/b3" to "Em/G", "C:dim7" to "C°7", "C:maj(9)" to "C",
        )
        for ((label, expected) in cases) assertEquals(label, expected, chordDisplayName(label))
    }

    /** Chords: Basic keeps only the triad family and drops the slash bass. */
    @Test fun basicSimplification() {
        val cases = listOf(
            "A:maj" to "A", "G:7" to "G", "C:maj7" to "C", "D:9" to "D", "F#:min7" to "F♯m",
            "A:min9" to "Am", "E:minmaj7" to "Em", "B:hdim7" to "B°", "C:dim7" to "C°",
            "C:aug" to "C+", "A:sus4(b7)" to "Asus4", "D:sus2" to "Dsus2", "D:maj/3" to "D", "N" to "",
        )
        for ((label, expected) in cases) assertEquals(label, expected, chordDisplayName(label, simplify = true))
    }

    @Test fun forcedAccidentals() {
        assertEquals("A♯", chordDisplayName("A#:maj", accidentals = Accidentals.SHARP))
        assertEquals("B♭", chordDisplayName("A#:maj", accidentals = Accidentals.FLAT))
        assertEquals("F♯m7", chordDisplayName("Gb:min7", accidentals = Accidentals.SHARP))
        assertEquals("D/G♭", chordDisplayName("D:maj/3", accidentals = Accidentals.FLAT))
    }

    @Test fun autoAccidentalsFollowTheKey() {
        val fMajor = Analysis.Key("F", Analysis.Mode.MAJOR)
        val dMajor = Analysis.Key("D", Analysis.Mode.MAJOR)
        val gMinor = Analysis.Key("G", Analysis.Mode.MINOR)
        val eMinor = Analysis.Key("E", Analysis.Mode.MINOR)
        val cMajor = Analysis.Key("C", Analysis.Mode.MAJOR)

        assertEquals("B♭", chordDisplayName("A#:maj", key = fMajor))
        assertEquals("E♭", chordDisplayName("D#:maj", key = gMinor))
        assertEquals("F♯m", chordDisplayName("F#:min", key = dMajor))
        assertEquals("D♯°", chordDisplayName("D#:dim", key = eMinor))
        assertEquals("C major uses the common spelling", "B♭", chordDisplayName("A#:maj", key = cMajor))
        assertEquals("C♯", chordDisplayName("C#:maj", key = cMajor))
        assertEquals("no key uses the common spelling", "B♭", chordDisplayName("A#:maj"))
    }

    @Test fun autoAccidentalsUseTheTransposedKey() {
        val fMajor = Analysis.Key("F", Analysis.Mode.MAJOR)
        // F major (flats) + 2 = G major (sharps): C# + 2 is spelled D♯, not E♭.
        assertEquals("D♯", chordDisplayName("C#:maj", transpose = 2, key = fMajor))
        assertEquals("D♭", chordDisplayName("C#:maj", transpose = 0, key = fMajor))
    }

    @Test fun transposeWrapsAround() {
        val cases = listOf(
            Triple("B:maj", 1, "C"),
            Triple("C:maj", -1, "B"),
            Triple("A:min7", 15, "Cm7"),
            Triple("A:min7", -9, "Cm7"),
            Triple("D:maj/3", 2, "E/G♯"),
            Triple("G:7", 12, "G7"),
            Triple("N", 3, ""),
        )
        for ((label, transpose, expected) in cases) {
            assertEquals("$label $transpose", expected, chordDisplayName(label, transpose, Accidentals.SHARP))
        }
    }

    @Test fun everyFixtureLabelHasAName() {
        val analysis = Contracts.sampleAnalysis()
        for (chord in analysis.chords.filterNot { isNoChord(it.label) }) {
            val name = chordDisplayName(chord.label, key = analysis.key)
            assertFalse(chord.label, name.isEmpty())
            assertFalse("${chord.label} → $name", name.contains(':'))
        }
    }
}
