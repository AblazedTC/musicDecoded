package com.musicdecoded.mobile.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.max
import kotlin.math.min

/**
 * WCAG 2.x contrast of the text tokens (T-503a): ≥ 4.5:1 for body text on every surface it's
 * drawn on. Measured DESIGN.md values that fail: `textMuted` #63778C (canvas 4.00, surface 3.60,
 * transport 3.33; lifted to #8396AA, T-506) and `record` #EC6478 on miniPlayerBg (4.45),
 * surfaceSoft (3.98) and primarySoft (3.76), where it isn't used for text.
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
    private val screens = arrayOf("ink" to c.Ink, "canvas" to c.Canvas, "surface" to c.Surface, "transportBg" to c.TransportBg, "timelineBg" to c.TimelineBg)
    private val raised = arrayOf("surfaceRaised" to c.SurfaceRaised, "miniPlayerBg" to c.MiniPlayerBg)

    @Test fun textTokensOnAllSurfaces() {
        assertReadable("text", c.Text, *screens, *raised, "surfaceSoft" to c.SurfaceSoft, "primarySoft" to c.PrimarySoft)
        assertReadable("textSecondary", c.TextSecondary, *screens, *raised, "surfaceSoft" to c.SurfaceSoft, "primarySoft" to c.PrimarySoft)
        assertReadable("primary", c.Primary, *screens, *raised, "surfaceSoft" to c.SurfaceSoft, "primarySoft" to c.PrimarySoft)
        assertReadable("secondary", c.Secondary, *screens, *raised, "surfaceSoft" to c.SurfaceSoft)
        assertReadable("chord", c.Chord, *screens, *raised, "chordMuted" to c.ChordMuted)
    }

    @Test fun mutedAndErrorTextOnTheSurfacesTheyUse() {
        assertReadable("textMuted", c.TextMuted, *screens, *raised)
        // Delete in menus (surfaceRaised), errors on cards and sheets.
        assertReadable("record", c.Record, *screens, "surfaceRaised" to c.SurfaceRaised)
    }

    @Test fun textOnFills() {
        assertReadable("onPrimary", c.OnPrimary, "primary" to c.Primary, "pianoBlackActive" to c.PianoBlackActive)
        assertReadable("onChord", c.OnChord, "chord" to c.Chord)
        assertReadable("pianoBlack", c.PianoBlack, "pianoWhite" to c.PianoWhite, "pianoWhiteActive" to c.PianoWhiteActive)
    }
}
