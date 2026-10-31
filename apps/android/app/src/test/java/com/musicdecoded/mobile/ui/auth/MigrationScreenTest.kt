package com.musicdecoded.mobile.ui.auth

import android.app.Application
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.hasContentDescription
import com.musicdecoded.mobile.ui.paywall.LocalOpenPaywall
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
class MigrationScreenTest {
    @get:Rule val compose = createComposeRule()

    private var paywallOpens = 0

    private fun show(pro: Boolean) {
        val summary = MigrationSummary(timelines = 3, ownAudio = 2, ownAudioBytes = 6_000_000, restricted = 1, ownAudioSyncs = pro)
        val vm = MigrationViewModel(load = { summary }, requestSync = {})
        compose.setContent {
            CompositionLocalProvider(LocalOpenPaywall provides { paywallOpens++ }) {
                MusicDecodedTheme { MigrationScreen(vm, onDone = {}) }
            }
        }
    }

    @Test fun freeAccountsSeeAudioAsProWithANoteThatOpensThePaywall() {
        show(pro = false)
        compose.onNode(hasContentDescription("2 local audio files", substring = true) and hasContentDescription("Pro only", substring = true))
            .assertExists()
        compose.onNodeWithText("PRO").assertExists()
        compose.onNodeWithText("part of Pro", substring = true).performScrollTo().assertHasClickAction().performClick()
        assertEquals(1, paywallOpens)
    }

    @Test fun proAccountsSyncTheirAudio() {
        show(pro = true)
        compose.onNodeWithText("Eligible for private sync · 6.0 MB", substring = true).assertExists()
        compose.onNodeWithText("part of Pro", substring = true).assertDoesNotExist()
    }
}
