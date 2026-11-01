package com.musicdecoded.mobile.player

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test fun fixedKeyboardIsC3ToB5() {
        assertEquals(0, keyboardWhiteIndex(48))
        assertEquals(0, keyboardWhiteIndex(49)) // C♯3 sits on C3
        assertEquals(20, keyboardWhiteIndex(83))
    }

    @Test fun focusKeysTakeTheSpaceLeft() {
        // Rest = 900 − 190 = 710, so an 800 viewport leaves 90 → clamped to the 96 minimum.
        assertEquals(96, fitKeysHeight(viewport = 800, content = 900, keys = 190, min = 96, max = 190))
        assertEquals(150, fitKeysHeight(viewport = 860, content = 900, keys = 190, min = 96, max = 190))
        // Plenty of room: never taller than the max.
        assertEquals(190, fitKeysHeight(viewport = 2000, content = 806, keys = 96, min = 96, max = 190))
        // Rounding jitter under 2px keeps the current height.
        assertEquals(150, fitKeysHeight(viewport = 861, content = 860, keys = 150, min = 96, max = 190))
    }

    @Test fun inversionNames() {
        assertEquals(listOf("Root", "1st inv", "2nd inv", "3rd inv"), (0..3).map(::inversionName))
    }

    @Test fun describesTheHighlightedKeys() {
        assertEquals("Piano, no chord", pianoDescription(0, null))
        assertEquals("Piano, keys E, G♯, B highlighted", pianoDescription(3, "E · G♯ · B"))
    }
}
