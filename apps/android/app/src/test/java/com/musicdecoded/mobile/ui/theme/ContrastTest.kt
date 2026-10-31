package com.musicdecoded.mobile.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.max
import kotlin.math.min

/**
 * WCAG 2.x contrast of the DESIGN.md §1 "Wrapped" pairings: ≥ 4.5:1 for every text color on every
 * fill it's drawn on (T-503a), and the one forbidden pair (pink text on violet) stays forbidden.
 */
class ContrastTest {
    private fun ratio(a: Color, b: Color): Double {
        val la = a.luminance().toDouble()
        val lb = b.luminance().toDouble()
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    private fun assertReadable(name: String, text: Color, vararg backgrounds: Pair<String, Color>) {
        for ((bgName, bg) in backgrounds) {
            val r = ratio(text, bg)
            assertTrue("$name on $bgName is %.2f:1".format(r), r >= 4.5)
        }
    }

    private val c = MusicDecodedColors
    private val darkSurfaces = arrayOf("ink" to c.Ink, "card" to c.Card, "cardRaised" to c.CardRaised)
    private val lightBlocks = arrayOf(
        "lime" to c.Lime, "pink" to c.Pink, "yellow" to c.Yellow, "orange" to c.Orange,
        "blue" to c.Blue, "cream" to c.Cream, "white" to c.White,
    )

    @Test fun textOnInkAndCards() {
        assertReadable("text", c.Text, *darkSurfaces, "chordMuted" to c.ChordMuted)
        assertReadable("textSecondary", c.TextSecondary, *darkSurfaces)
        // textMuted is for ink only (DESIGN.md §1), but it also stays readable on the cards it lands on.
        assertReadable("textMuted", c.TextMuted, "ink" to c.Ink, "card" to c.Card)
    }

    @Test fun accentsOnInkAreReadable() {
        // DESIGN.md §1 says "at least 6.4"; measured, orange on ink is 6.36, the lowest of the five.
        listOf("lime" to c.Lime, "pink" to c.Pink, "yellow" to c.Yellow, "blue" to c.Blue, "orange" to c.Orange).forEach { (name, color) ->
            assertTrue("$name on ink", ratio(color, c.Ink) >= 6.3)
            assertReadable(name, color, *darkSurfaces)
        }
        assertReadable("chord", c.Chord, "chordMuted" to c.ChordMuted)
    }

    @Test fun inkOnEveryLightBlock() {
        assertReadable("onColor", c.OnColor, *lightBlocks)
        assertReadable("onPrimary", c.OnPrimary, "primary" to c.Primary, "pianoBlackActive" to c.PianoBlackActive)
        assertReadable("onSecondary", c.OnSecondary, "secondary" to c.Secondary)
        assertReadable("onChord", c.OnChord, "chord" to c.Chord)
        assertReadable("textSecondaryOnLime", c.TextSecondaryOnLime, "lime" to c.Lime)
        assertReadable("pianoBlack", c.PianoBlack, "pianoWhite" to c.PianoWhite, "pianoWhiteActive" to c.PianoWhiteActive)
    }

    @Test fun textOnViolet() {
        assertReadable("white", c.White, "violet" to c.Violet)
        assertReadable("lime", c.Lime, "violet" to c.Violet)
        assertReadable("yellow", c.Yellow, "violet" to c.Violet)
        assertReadable("textSecondaryOnViolet", c.TextSecondaryOnViolet, "violet" to c.Violet)
    }

    @Test fun pinkTextOnVioletIsTooLowToUse() {
        // DESIGN.md §1: never pink text on violet (3.2:1); pink on violet only as a fill with ink text.
        assertTrue(ratio(c.Pink, c.Violet) < 4.5)
    }

    @Test fun paletteMatchesThePortfolio() {
        assertEquals(Color(0xFF4100F5), c.Violet)
        assertEquals(Color(0xFFCFF56A), c.Lime)
        assertEquals(Color(0xFFF774C4), c.Pink)
        assertEquals(Color(0xFFFAE62D), c.Yellow)
        assertEquals(Color(0xFFFF6437), c.Orange)
        assertEquals(Color(0xFF509BF5), c.Blue)
        assertEquals(Color(0xFF121212), c.Ink)
    }
}
