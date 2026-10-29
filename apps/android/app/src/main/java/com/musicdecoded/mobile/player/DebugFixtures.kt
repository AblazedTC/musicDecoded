package com.musicdecoded.mobile.player

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.ChordVoicings

/**
 * The contract fixtures, packaged as assets in debug builds only (see `app/build.gradle.kts`),
 * so the player can be tried and previewed without the backend.
 */
object DebugFixtures {
    /** `meta.trackId` of `contracts/fixtures/sunnydays.analysis.json` (has real waveform peaks). */
    const val SUNNY_DAYS = "fixture-sunnydays"

    /** Loads `fixtures/<name>.analysis.json` for a `fixture-<name>` id; `null` if not packaged. */
    fun load(context: Context, trackId: String): Analysis? {
        if (!trackId.startsWith(PREFIX)) return null
        val path = "fixtures/${trackId.removePrefix(PREFIX)}.analysis.json"
        return runCatching { context.assets.open(path).use { Analysis.decode(it.reader().readText()) } }.getOrNull()
    }

    private const val PREFIX = "fixture-"
}

/** `contracts/chord-voicings.json`, packaged as an asset in every build. */
fun loadChordVoicings(context: Context): ChordVoicings =
    context.assets.open("chord-voicings.json").use { ChordVoicings.decode(it.reader().readText()) }

/** The Sunny Days fixture for `@Preview`s (`null` if the assets aren't available). */
@Composable
internal fun rememberPreviewAnalysis(): Analysis? {
    val context = LocalContext.current
    return remember { DebugFixtures.load(context, DebugFixtures.SUNNY_DAYS) }
}
