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
import androidx.compose.ui.draw.alpha
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
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
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
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        BrandHeader(onSearch = nav.onSearch, onProfile = nav.onProfile)

        Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            SectionHeader("Recently Decoded", actionLabel = "See all", onAction = nav.onSeeAll)
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

        Column(verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            SectionHeader("Transcribe Music")
            SourceGrid(soundcloudEnabled = state.soundcloudEnabled, nav = nav)
            RecordCard(onClick = nav.onRecord)
        }

        TipCard()
    }
}

@Composable
private fun BrandHeader(onSearch: () -> Unit, onProfile: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.sm)) {
        Box(
            Modifier.size(Sizes.iconButton).rotate(LOGO_TILT).background(MusicDecodedColors.Primary, Shapes.control),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Rounded.GraphicEq, contentDescription = null, tint = MusicDecodedColors.OnPrimary)
        }
        Column(Modifier.weight(1f)) {
            Text("MusicDecoded", style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Text)
            Text("Your music, decoded.", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        MdIconButton(Icons.Rounded.Search, contentDescription = "Search your library", onClick = onSearch)
        MdIconButton(Icons.Rounded.Person, contentDescription = "Profile and settings", onClick = onProfile)
    }
}

private const val LOGO_TILT = -4f

/** 2×2 grid: Files · Media Library · YouTube · SoundCloud (only when enabled; its flow is still M3). */
@Composable
private fun SourceGrid(soundcloudEnabled: Boolean, nav: HomeNavigation) {
    val cards = buildList<@Composable (Modifier) -> Unit> {
        add { SourceCard(SourceKind.FILES, "Files", "Import audio from this device", nav.onFiles, it) }
        add { SourceCard(SourceKind.MEDIA_LIBRARY, "Media Library", "Choose local music", nav.onMediaLibrary, it) }
        add { SourceCard(SourceKind.YOUTUBE, "YouTube", "Discover & play", nav.onYouTube, it) }
        // Only when the server enables it; the SoundCloud flow itself arrives in M3.
        if (soundcloudEnabled) {
            add { SourceCard(SourceKind.SOUNDCLOUD, "SoundCloud", "Coming soon", onClick = {}, modifier = it.alpha(DISABLED_ALPHA)) }
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
        cards.chunked(2).forEach { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                pair.forEach { card -> card(Modifier.weight(1f)) }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
    }
}

private const val DISABLED_ALPHA = 0.5f

@Composable
private fun RecordCard(onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clip(Shapes.card)
            .background(MusicDecodedColors.Surface)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier.size(Sizes.sourceTile).background(SourceKind.RECORD.background, Shapes.control),
            contentAlignment = Alignment.Center,
        ) {
            Icon(SourceKind.RECORD.icon, contentDescription = null, tint = SourceKind.RECORD.tint)
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text("Record Audio", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            Text("Capture a live performance", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = MusicDecodedColors.TextMuted)
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
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Icon(Icons.Rounded.AutoAwesome, contentDescription = null, tint = MusicDecodedColors.Chord)
        Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Chord)
            Text(body, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
    }
}

@Preview
@Composable
private fun HomeSectionsPreview() = PreviewSurface {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.lg)) {
        BrandHeader(onSearch = {}, onProfile = {})
        RecordCard(onClick = {})
        TipCard()
    }
}
