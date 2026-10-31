package com.musicdecoded.mobile.ui.library

import android.app.Application
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.getUnclippedBoundsInRoot
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.hasScrollAction
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Spacing
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class LibraryFilterRowTest {
    @get:Rule val compose = createComposeRule()

    @Test fun lastChipScrollsClearOfTheScreenEdge() {
        compose.setContent {
            MusicDecodedTheme { Box(Modifier.width(280.dp).testTag("screen")) { LibraryFilterRow(LibraryFilter.ALL, onSelect = {}) } }
        }
        val rootRight = compose.onNodeWithTag("screen").getUnclippedBoundsInRoot().right
        // Narrow screen: RECORDINGS starts past the edge; scrolled to the end, the gutter follows it.
        assertTrue(compose.onNodeWithText("RECORDINGS").getUnclippedBoundsInRoot().right > rootRight)
        compose.onNode(hasScrollAction()).performSemanticsAction(SemanticsActions.ScrollBy) { it(10_000f, 0f) }
        compose.waitForIdle()
        val chipRight = compose.onNodeWithText("RECORDINGS").getUnclippedBoundsInRoot().right
        assertTrue("chip right $chipRight vs root $rootRight", chipRight <= rootRight - Spacing.gutter + 0.5.dp)
    }
}
