package com.musicdecoded.mobile.sources.youtube

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.network.VideoSummary
import com.musicdecoded.mobile.player.YouTubeWebPlayer
import com.musicdecoded.mobile.player.youTubeErrorMessage
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PolicyNote
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.RemoteImage
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing

/** DESIGN.md §3.4: the hero is 202 tall. */
private val HeroHeight = 202.dp

/** PLAN.md R-1, verbatim (DESIGN.md §3.4: "This copy *is* PLAN R-1. Keep it."). */
const val YOUTUBE_POLICY_NOTE = "Offline audio is saved only when permitted. The chord timeline can remain available separately."

/** Video detail (DESIGN.md §3.4). */
@Composable
fun YouTubeDetailScreen(
    video: VideoSummary,
    onBack: () -> Unit,
    onDecodeStarted: (trackId: String) -> Unit,
    onOpenPlayer: (trackId: String) -> Unit,
    onOpenProcessing: (trackId: String) -> Unit,
) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel: YouTubeDetailViewModel = viewModel(key = "youtube-${video.videoId}", factory = YouTubeDetailViewModel.factory(app, video))
    val state by viewModel.state.collectAsStateWithLifecycle()

    LaunchedEffect(state.startedTrackId) {
        state.startedTrackId?.let {
            viewModel.navigated()
            onDecodeStarted(it)
        }
    }

    YouTubeDetailContent(
        state,
        onBack = onBack,
        onPlay = viewModel::play,
        onDecode = viewModel::decode,
        onOpenExisting = {
            val existing = state.existing ?: return@YouTubeDetailContent
            if (existing.status == LibraryStatus.READY) onOpenPlayer(existing.trackId) else onOpenProcessing(existing.trackId)
        },
    )
}

@Composable
private fun YouTubeDetailContent(
    state: YouTubeDetailUiState,
    onBack: () -> Unit,
    onPlay: () -> Unit,
    onDecode: () -> Unit,
    onOpenExisting: () -> Unit,
) {
    val video = state.video
    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        Row(Modifier.padding(horizontal = Spacing.gutter, vertical = Spacing.xs)) {
            MdIconButton(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "Back", onClick = onBack)
        }
        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            Hero(video, playing = state.playing, onPlay = onPlay)
            Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                Text(video.title, style = MaterialTheme.typography.titleLarge, color = MusicDecodedColors.Text)
                if (video.channelTitle.isNotEmpty()) {
                    Text(video.channelTitle, style = MaterialTheme.typography.bodyMedium, color = MusicDecodedColors.TextSecondary)
                }
            }
            AvailabilityCard(state)
            val existing = state.existing
            when {
                existing?.status == LibraryStatus.READY -> PrimaryButton("Open transcription", onClick = onOpenExisting)
                existing?.status == LibraryStatus.ANALYZING -> PrimaryButton("View progress", onClick = onOpenExisting)
                else -> PrimaryButton(
                    if (state.starting) "Starting…" else "Decode this song",
                    onClick = onDecode,
                    icon = Icons.Rounded.AutoAwesome,
                    enabled = !state.starting,
                )
            }
            state.error?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.Record) }
            PolicyNote(YOUTUBE_POLICY_NOTE, icon = Icons.Rounded.Shield)
        }
    }
}

/** Thumbnail with a play button; playing swaps in the official embed, inline. */
@Composable
private fun Hero(video: VideoSummary, playing: Boolean, onPlay: () -> Unit) {
    var error by remember(video.videoId) { mutableStateOf<String?>(null) }
    Box(Modifier.fillMaxWidth().height(HeroHeight).clip(Shapes.card).background(MusicDecodedColors.Ink)) {
        if (playing) {
            YouTubeWebPlayer(video.videoId, Modifier.matchParentSize(), autoplay = true, onError = { error = youTubeErrorMessage(it) })
            error?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.Text,
                    modifier = Modifier.align(Alignment.Center).padding(Spacing.md),
                )
            }
        } else {
            RemoteImage(video.thumbnailUrl, Modifier.matchParentSize())
            Box(
                Modifier
                    .align(Alignment.Center)
                    .size(Sizes.playButton)
                    .clip(CircleShape)
                    .background(MusicDecodedColors.Ink.copy(alpha = 0.7f))
                    .clickable(role = Role.Button, onClickLabel = "Play video", onClick = onPlay),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Rounded.PlayArrow, contentDescription = "Play video", tint = MusicDecodedColors.Text, modifier = Modifier.size(Sizes.icon))
            }
            video.durationSec?.let { DurationBadge(it, Modifier.align(Alignment.BottomEnd).padding(Spacing.xs)) }
        }
    }
}

@Composable
private fun AvailabilityCard(state: YouTubeDetailUiState) {
    val (title, body) = when (state.existing?.status) {
        LibraryStatus.READY -> "Already decoded" to "This song's chord timeline is in your library."
        LibraryStatus.ANALYZING -> "Decoding" to "This song is being decoded now."
        else -> "Analysis available" to "Decode chords, key and tempo, then play along with the video."
    }
    Row(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Rounded.CheckCircle, contentDescription = null, tint = MusicDecodedColors.Primary)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            Text(body, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
    }
}

@Preview
@Composable
private fun YouTubeDetailPreview() = PreviewSurface {
    YouTubeDetailContent(
        YouTubeDetailUiState(VideoSummary("abcdefghijk", "Sunny Days (Official Video)", "Some Channel", "", 214)),
        {}, {}, {}, {},
    )
}
