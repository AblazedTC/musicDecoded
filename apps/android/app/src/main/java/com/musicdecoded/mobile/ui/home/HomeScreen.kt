package com.musicdecoded.mobile.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.SectionHeader
import com.musicdecoded.mobile.ui.components.SourceCard
import com.musicdecoded.mobile.ui.components.SourceKind
import com.musicdecoded.mobile.ui.library.LibrarySongRow
import com.musicdecoded.mobile.ui.library.SongListMessages
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.ui.components.Eyebrow
import com.musicdecoded.mobile.ui.components.SourceIconTile
import com.musicdecoded.mobile.ui.components.Sticker
import com.musicdecoded.mobile.ui.components.WaveformBars
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.halftone
import com.musicdecoded.mobile.ui.components.tilt
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Tilt
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlinx.coroutines.delay

/** Where Home's cards and buttons lead (wired to the nav graph in MainScreen). */
class HomeNavigation(
    val onSeeAll: () -> Unit,
    val onSearch: () -> Unit,
    val onProfile: () -> Unit,
    val onFiles: () -> Unit,
    val onMediaLibrary: () -> Unit,
    val onYouTube: () -> Unit,
    val onSoundCloud: () -> Unit,
    val onRecord: () -> Unit,
    val onOpenPlayer: (trackId: String) -> Unit,
    val onOpenProcessing: (trackId: String) -> Unit,
)

/** DESIGN.md §3.1. */
@Composable
fun HomeScreen(nav: HomeNavigation, modifier: Modifier = Modifier) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel: HomeViewModel = viewModel(factory = HomeViewModel.factory(app))
    val state by viewModel.state.collectAsStateWithLifecycle()
    val message by viewModel.message.collectAsStateWithLifecycle()
    SongListMessages(message, viewModel::messageShown)

    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(Spacing.xl),
    ) {
        BrandHeader(onSearch = nav.onSearch, onProfile = nav.onProfile)

        Column(
            Modifier.padding(horizontal = Spacing.gutter).padding(bottom = Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.xl),
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                SectionHeader("Recently Decoded", actionLabel = "See all", onAction = nav.onSeeAll, numeral = "01")
                if (!state.loading && state.recent.isEmpty()) {
                    Text(
                        "Nothing decoded yet. Pick a source below to decode your first song.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MusicDecodedColors.TextSecondary,
                    )
                }
                state.recent.forEach { row ->
                    LibrarySongRow(row, viewModel, nav.onOpenPlayer, nav.onOpenProcessing)
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
                SectionHeader("Transcribe Music", numeral = "02")
                SourceGrid(soundcloudEnabled = state.soundcloudEnabled, nav = nav)
                RecordCard(onClick = nav.onRecord)
            }

            TipCard()
        }
    }
}

/**
 * The violet hero block (DESIGN.md §1 color map): halftone texture, logo sticker, search/profile,
 * the "MUSIC / DECODED" display wordmark, tagline and lime waveform bars.
 */
@Composable
private fun BrandHeader(onSearch: () -> Unit, onProfile: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Violet)
            .halftone(MusicDecodedColors.White)
            .padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            Box(
                Modifier.size(Sizes.iconButton).tilt(LOGO_TILT).block(MusicDecodedColors.Primary, shadow = Shadows.sm),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Rounded.GraphicEq, contentDescription = null, tint = MusicDecodedColors.OnPrimary)
            }
            Sticker("Chord decoder", Modifier.padding(start = Spacing.xs), fill = MusicDecodedColors.Pink)
            Spacer(Modifier.weight(1f))
            MdIconButton(Icons.Rounded.Search, contentDescription = "Search your library", onClick = onSearch, container = MusicDecodedColors.White)
            MdIconButton(Icons.Rounded.Person, contentDescription = "Profile and settings", onClick = onProfile, container = MusicDecodedColors.White)
        }
        Column(Modifier.semantics(mergeDescendants = true) { heading() }) {
            Text("MUSIC", style = MaterialTheme.typography.headlineMedium.copy(fontSize = WordmarkSize), color = MusicDecodedColors.Text)
            Text("DECODED", style = MaterialTheme.typography.headlineMedium.copy(fontSize = WordmarkSize), color = MusicDecodedColors.Lime)
        }
        Text("Your music, decoded.", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
        WaveformBars(Modifier.fillMaxWidth().height(Spacing.xxxl), color = MusicDecodedColors.Lime)
    }
}

private const val LOGO_TILT = -4f

/** The Home wordmark is bigger than a page title (hero only). */
private val WordmarkSize = 56.sp

/** 2×2 grid: Files · Media Library · YouTube · SoundCloud (only when the server enables it). */
@Composable
private fun SourceGrid(soundcloudEnabled: Boolean, nav: HomeNavigation) {
    val cards = buildList<@Composable (Modifier) -> Unit> {
        // List cards alternate -1°/+1° (DESIGN.md §1 tilt).
        add { SourceCard(SourceKind.FILES, "Files", "Import audio from this device", nav.onFiles, it, tilt = -Tilt.listCard) }
        add { SourceCard(SourceKind.MEDIA_LIBRARY, "Media Library", "Choose local music", nav.onMediaLibrary, it, tilt = Tilt.listCard) }
        add { SourceCard(SourceKind.YOUTUBE, "YouTube", "Discover & play", nav.onYouTube, it, tilt = Tilt.listCard) }
        if (soundcloudEnabled) {
            add { SourceCard(SourceKind.SOUNDCLOUD, "SoundCloud", "Find supported tracks", nav.onSoundCloud, it, tilt = -Tilt.listCard) }
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
        cards.chunked(2).forEach { pair ->
            Row(Modifier.height(IntrinsicSize.Max), horizontalArrangement = Arrangement.spacedBy(Spacing.md)) {
                pair.forEach { card -> card(Modifier.weight(1f).fillMaxHeight()) }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun RecordCard(onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .block(SourceKind.RECORD.fill, shadow = Shadows.md, shadowColor = MusicDecodedColors.Violet, onClick = onClick)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        SourceIconTile(SourceKind.RECORD, fill = MusicDecodedColors.White)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text("RECORD AUDIO", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.OnColor)
            Text("Capture a live performance", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.OnColor)
        }
        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = MusicDecodedColors.OnColor)
    }
}

private val TIPS = listOf(
    "Practice smarter" to "Slow down any decoded song without changing its key.",
    "Loop a tricky bar" to "Tap loop in the player to repeat the current bar until it clicks.",
    "Change the key" to "Transpose the chords up or down to suit your voice or capo.",
    "Works offline" to "Decoded songs with the download badge play without a connection.",
)

private const val TIP_ROTATE_MS = 8_000L

/** Rotating tips (sparkle icon, chord color). */
@Composable
private fun TipCard() {
    var index by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(TIP_ROTATE_MS)
            index = (index + 1) % TIPS.size
        }
    }
    val (title, body) = TIPS[index]
    Row(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface)
            .border(Borders.thick, MusicDecodedColors.BorderOnInk)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Icon(Icons.Rounded.AutoAwesome, contentDescription = null, tint = MusicDecodedColors.Chord)
        Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Eyebrow("Tip")
            Text(title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Chord)
            Text(body, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
    }
}

@Preview
@Composable
private fun HomeSectionsPreview() = MusicDecodedTheme {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.lg)) {
        BrandHeader(onSearch = {}, onProfile = {})
        RecordCard(onClick = {})
        TipCard()
    }
}
