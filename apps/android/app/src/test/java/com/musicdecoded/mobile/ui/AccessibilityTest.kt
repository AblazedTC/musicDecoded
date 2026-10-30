package com.musicdecoded.mobile.ui

import android.app.Application
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.isHeading
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.player.PianoKeyboard
import com.musicdecoded.mobile.player.SeekBar
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.ErrorState
import com.musicdecoded.mobile.ui.components.InfoChip
import com.musicdecoded.mobile.ui.components.MetaChip
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.components.SectionHeader
import com.musicdecoded.mobile.ui.components.SongRow
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** TalkBack semantics and font scaling of the shared components (T-503a). */
@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE) // real text measurement for the font-scaling check
class AccessibilityTest {
    @get:Rule val compose = createComposeRule()

    private fun setContent(fontScale: Float = 1f, content: @androidx.compose.runtime.Composable () -> Unit) = compose.setContent {
        val density = LocalDensity.current
        CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale)) {
            MusicDecodedTheme(content)
        }
    }

    @Test fun headersAreHeadings() {
        setContent {
            PageHeader("Library", overline = "Your collection")
            SectionHeader("Recently Decoded", actionLabel = "See all", onAction = {})
        }
        // Display titles are uppercase (DESIGN.md §1).
        compose.onNode(isHeading() and hasText("LIBRARY")).assertExists()
        compose.onNode(isHeading() and hasText("RECENTLY DECODED")).assertExists()
    }

    @Test fun songRowIsOneButtonWithALabelledMoreMenu() {
        var opened = 0
        var more = 0
        setContent {
            SongRow("Sunny Days", "Artist", "4:23 · D major", onClick = { opened++ }, onMore = { more++ }, offline = true)
        }
        // Title, artist, meta and the offline badge are read as one row.
        compose.onNode(hasText("Sunny Days") and hasText("Artist") and hasText("4:23 · D major"))
            .assertHasClickAction()
            .assert(SemanticsMatcher.expectValue(SemanticsProperties.ContentDescription, listOf("Available offline")))
            .performClick()
        compose.onNodeWithContentDescription("More options for Sunny Days").assertHasClickAction().performClick()
        assertEquals(1, opened)
        assertEquals(1, more)
    }

    @Test fun errorStateOffersRetry() {
        var retried = 0
        setContent { ErrorState(TranscriptionRepository.OFFLINE, onRetry = { retried++ }) }
        compose.onNodeWithText("You're offline").assertExists()
        compose.onNodeWithText("Try again").assertHasClickAction().performClick()
        assertEquals(1, retried)
    }

    @Test fun pianoReadsTheHighlightedNotes() {
        setContent { PianoKeyboard(notes = listOf(59, 62, 66), description = "B · D · F♯") }
        compose.onNodeWithContentDescription("Piano, keys B, D, F♯ highlighted").assertExists()
    }

    @Test fun seekBarSaysWhereItIs() {
        val position = mutableDoubleStateOf(16.0)
        setContent { SeekBar(position, 263.0, onSeek = {}) }
        compose.onNodeWithContentDescription("Seek")
            .assert(SemanticsMatcher.expectValue(SemanticsProperties.StateDescription, "0:16 of 4:23"))
    }

    @Test fun chipsAreReadAsOneAndDontClipAtDoubleFontSize() {
        setContent(fontScale = 2f) {
            androidx.compose.foundation.layout.Column {
                InfoChip(Icons.Rounded.MusicNote, "Key", "F♯ minor", Modifier.width(110.dp))
                ControlChip("Transpose", "+6", onClick = {}, Modifier.width(110.dp))
            }
        }
        compose.onNode(hasText("KEY") and hasText("F♯ minor")).assertExists()
        listOf("F♯ minor", "+6", "KEY", "TRANSPOSE").forEach { assertFullyShown(it) }
    }

    @Test fun compactPlayerChipsReadTheirLabel() {
        setContent {
            androidx.compose.foundation.layout.Column {
                MetaChip(Icons.Rounded.MusicNote, "Key", "D major")
                ControlChip("Chords", "Full", onClick = {}, compact = true)
            }
        }
        compose.onNodeWithContentDescription("Key, D major").assertExists()
        compose.onNode(hasText("CHORDS") and hasText("Full")).assertHasClickAction()
    }

    @Test fun removeAnimationsSettingTurnsOffTilt() {
        val context = androidx.test.core.app.ApplicationProvider.getApplicationContext<Application>()
        android.provider.Settings.Global.putFloat(context.contentResolver, android.provider.Settings.Global.ANIMATOR_DURATION_SCALE, 0f)
        var reduceMotion: Boolean? = null
        setContent { reduceMotion = com.musicdecoded.mobile.ui.theme.LocalReduceMotion.current }
        compose.waitForIdle()
        assertEquals(true, reduceMotion)
    }

    /** Every character is laid out (wrapping is fine): no height clip, no ellipsis, no dropped tail. */
    private fun assertFullyShown(text: String) {
        val layout = compose.onNodeWithText(text, useUnmergedTree = true).textLayout()
        assertFalse("$text clipped vertically", layout.didOverflowHeight)
        val last = layout.lineCount - 1
        assertFalse("$text ellipsized", layout.isLineEllipsized(last))
        assertEquals("$text cut off", text.length, layout.getLineEnd(last))
    }

    private fun SemanticsNodeInteraction.textLayout(): TextLayoutResult {
        val results = mutableListOf<TextLayoutResult>()
        fetchSemanticsNode().config[SemanticsActions.GetTextLayoutResult].action!!.invoke(results)
        return results.single()
    }
}
