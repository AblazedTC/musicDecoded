package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Spacing

/**
 * Values from `contracts/fixtures/sample.analysis.json`, so previews show real fixture data
 * (the fixture has no title/artist; those are placeholders).
 */
internal object PreviewFixture {
    const val TITLE = "Sample track"
    const val ARTIST = "Fixture artist"
    const val DURATION_SEC = 263.407
    val KEY = Analysis.Key("D", Analysis.Mode.MAJOR)
    const val BPM = 136.1
    const val TIME_SIGNATURE = "4/4"
    const val CURRENT_CHORD = "D:maj7"
}

@Composable
internal fun PreviewSurface(content: @Composable () -> Unit) {
    MusicDecodedTheme {
        Box(Modifier.background(MusicDecodedColors.Canvas).padding(Spacing.md)) { content() }
    }
}
