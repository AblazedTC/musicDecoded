package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Contracts
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PianoKeyboardTest {
    @Test fun blackKeys() {
        assertEquals(listOf(1, 3, 6, 8, 10), (0..11).filter(::isBlackKey))
        assertTrue(isBlackKey(61))
        assertFalse(isBlackKey(60))
    }

    @Test fun whiteKeyIndexRoundTrips() {
        assertEquals(35, whiteKeyIndex(60)) // C4 = 5 octaves × 7
        assertEquals(35, whiteKeyIndex(61)) // C♯ sits on C's white key
        assertEquals(41, whiteKeyIndex(71)) // B4
        for (index in 0..74) assertEquals(index, whiteKeyIndex(midiOfWhiteKey(index)))
        assertEquals(60, midiOfWhiteKey(35))
        assertEquals(65, midiOfWhiteKey(38))
    }

    @Test fun startsAtTheCBelowTheLowestNote() {
        assertEquals(48, keyboardStart(listOf(50, 54, 57), octaves = 2)) // D3 F♯3 A3
        assertEquals(36, keyboardStart(listOf(42, 50, 54, 57), octaves = 2)) // D/F♯ fits from C2
    }

    @Test fun dropsTheBassWhenTheChordWontFit() {
        // Bass C2 + chord up to E4: 2 octaves can't hold both, so the chord tones win.
        assertEquals(48, keyboardStart(listOf(36, 48, 52, 64), octaves = 2))
        assertEquals(36, keyboardStart(listOf(36, 48, 52, 64), octaves = 3))
    }

    @Test fun noNotesHasNoStart() {
        assertNull(keyboardStart(emptyList(), octaves = 2))
        assertNull(centeredFirstWhite(emptyList(), octaves = 2))
    }

    @Test fun fittedKeyboardCentersTheActiveNotes() {
        // C4 E4 G4 = white keys 35..39 → 14-key window 31..44 (4 left, 5 right).
        assertEquals(31, centeredFirstWhite(listOf(60, 64, 67), octaves = 2))
        assertEquals(27, centeredFirstWhite(listOf(60, 64, 67), octaves = 3))
        // Clamped to the 88 keys.
        assertEquals(whiteKeyIndex(21), centeredFirstWhite(listOf(21, 24), octaves = 3))
    }

    @Test fun fittedOctavesFollowTheWidthButNeverFewerThanTwo() {
        assertEquals(2, fittedOctaves(width = 380f, height = 200f)) // 40-wide keys → 9 keys, min 2
        assertEquals(2, fittedOctaves(width = 380f, height = 96f)) // 19.2-wide → 19 keys
        assertEquals(4, fittedOctaves(width = 600f, height = 96f))
    }

    @Test fun everyVoicingExampleFitsTwoOctavesWithoutItsBass() {
        val voicings = Contracts.voicings()
        for ((label, notes) in voicings.examples.orEmpty().filterValues { it.isNotEmpty() }) {
            val start = checkNotNull(keyboardStart(notes, octaves = 2)) { label }
            val chordTones = if (label.contains('/')) notes.drop(1) else notes
            assertTrue(label, chordTones.all { it in start until start + 24 })
        }
    }
}
