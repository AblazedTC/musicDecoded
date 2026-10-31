package com.musicdecoded.mobile.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class ChordVoicingsTest {
    private val voicings = Contracts.voicings()

    @Test fun matchesEveryExampleInTheContract() {
        val examples = checkNotNull(voicings.examples)
        assertFalse(examples.isEmpty())
        for ((label, expected) in examples) {
            assertEquals(label, expected, voicings.notes(label))
        }
    }

    @Test fun followsTheAlgorithm() {
        val cases = listOf(
            "C:maj" to listOf(48, 52, 55),
            "Bb:7" to listOf(58, 62, 65, 68),
            "E:min/b3" to listOf(43, 52, 55, 59),
            "A:sus4(b7)" to listOf(57, 62, 64, 67),
            "C:min(9)" to listOf(48, 51, 55), // unknown → text before "(" → min
            "C:weird" to listOf(48, 52, 55), // unknown → maj
            "X" to emptyList(),
            "N" to emptyList(),
        )
        for ((label, expected) in cases) {
            assertEquals(label, expected, voicings.notes(label))
        }
    }

    @Test fun transposeShiftsThePitchClassAndWraps() {
        assertEquals(voicings.notes("C:maj"), voicings.notes("A:maj", transpose = 3))
        assertEquals(listOf(48, 52, 55), voicings.notes("B:maj", transpose = 1))
        assertEquals(listOf(59, 63, 66), voicings.notes("C:maj", transpose = -1))
        assertEquals(voicings.notes("F#:maj"), voicings.notes("A:maj", transpose = -15))
        assertEquals(listOf(44, 52, 56, 59), voicings.notes("D:maj/3", transpose = 2))
    }
}
