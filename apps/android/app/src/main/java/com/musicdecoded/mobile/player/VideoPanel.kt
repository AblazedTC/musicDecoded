package com.musicdecoded.mobile.player

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.VerifiedUser
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

private const val VIDEO_ASPECT = 16f / 9f
private const val PAUSED_SYNC_MS = 250L

/**
 * The Video visualizer (DESIGN.md §3.7 item 6): the official embed, 16:9, with the "Authorized
 * embedded playback" badge. Without local audio the video is the clock ([PlayerViewModel.videoEngine]);
 * with local audio a muted follower is kept within 150 ms of it ([VideoSync]).
 */
@Composable
internal fun PlayerVideoPanel(viewModel: PlayerViewModel, videoId: String, durationSec: Double, modifier: Modifier = Modifier) {
    val clock = viewModel.videoEngine
    if (clock != null) {
        DisposableEffect(clock, viewModel) {
            // A tap on the embed's own play/pause must reach the player state (and start the clock).
            clock.onPlayStateChange = viewModel::tick
            onDispose { clock.onPlayStateChange = null }
        }
        YouTubeVideo(videoId, clock, controls = true, modifier)
    } else {
        val follower = remember(videoId) { YouTubePlaybackEngine(durationSec, muted = true) }
        val sync = remember(follower) { VideoSync(follower) }
        DisposableEffect(follower) { onDispose { follower.release() } }
        LaunchedEffect(sync, viewModel) {
            while (isActive) {
                val state = viewModel.state.value
                sync.sync(viewModel.position.value, state.isPlaying, state.speed)
                if (state.isPlaying) withFrameNanos {} else delay(PAUSED_SYNC_MS)
            }
        }
        // The transport drives a follower; the embed's own controls would only fight it.
        YouTubeVideo(videoId, follower, controls = false, modifier)
    }
}

@Composable
private fun YouTubeVideo(videoId: String, engine: YouTubePlaybackEngine, controls: Boolean, modifier: Modifier = Modifier) {
    var error by remember(videoId) { mutableStateOf<String?>(null) }
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Box(Modifier.fillMaxWidth().aspectRatio(VIDEO_ASPECT).clip(Shapes.chip).background(MusicDecodedColors.Ink)) {
            YouTubeWebPlayer(
                videoId,
                Modifier.matchParentSize(),
                startSeconds = engine.position(),
                controls = controls,
                onError = { error = youTubeErrorMessage(it) },
                onAttach = engine::attach,
                onDetach = engine::detach,
            )
            error?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.Text,
                    modifier = Modifier.align(Alignment.Center).padding(Spacing.md),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xxs), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.VerifiedUser, contentDescription = null, tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
            Text("Authorized embedded playback", style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextSecondary)
        }
    }
}
