package com.musicdecoded.mobile.player

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PreviewFixture
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.SongArt
import com.musicdecoded.mobile.ui.components.keyName
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlinx.coroutines.delay

/** DESIGN.md §2: the 2px progress line on top of the mini player. */
private val ProgressLine = 2.dp
private const val TICK_MS = 250L

/**
 * Mini player above the tab bar (Home/Library) while [NowPlaying] has a track: art (initial),
 * title, "artist · key", play/pause, dismiss and a progress line. Tapping it opens the player.
 */
@Composable
fun MiniPlayer(
    state: NowPlaying.State,
    position: () -> Double,
    onTick: () -> Unit,
    onOpen: () -> Unit,
    onTogglePlay: () -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    // Keeps the progress line and play/pause icon in sync while the player screen isn't driving the clock.
    LaunchedEffect(state.trackId) {
        while (true) {
            onTick()
            delay(TICK_MS)
        }
    }
    Column(
        modifier
            .fillMaxWidth()
            .padding(start = Spacing.xs, end = Spacing.sm, top = Spacing.xxs, bottom = Spacing.xs)
            .block(MusicDecodedColors.Surface, shadow = Shadows.sm, shadowColor = MusicDecodedColors.Secondary, onClick = onOpen, onClickLabel = "Open player"),
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(ProgressLine)
                .drawBehind {
                    drawRect(MusicDecodedColors.SurfaceRaised)
                    val fraction = if (state.durationSec > 0) (position() / state.durationSec).toFloat().coerceIn(0f, 1f) else 0f
                    drawRect(MusicDecodedColors.Primary, size = Size(size.width * fraction, size.height))
                },
        )
        Row(
            Modifier.padding(start = Spacing.xs),
            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            SongArt(state.title, Modifier.size(Sizes.iconButton), style = MaterialTheme.typography.titleSmall)
            Column(Modifier.weight(1f)) {
                Text(state.title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                val subtitle = listOfNotNull(state.artist, state.key?.let(::keyName)).joinToString(" · ")
                if (subtitle.isNotEmpty()) {
                    Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
            MdIconButton(
                if (state.isPlaying) Icons.Rounded.Pause else Icons.Rounded.PlayArrow,
                contentDescription = if (state.isPlaying) "Pause" else "Play",
                onClick = onTogglePlay,
                container = MusicDecodedColors.Secondary,
            )
            MdIconButton(Icons.Rounded.Close, contentDescription = "Close player", onClick = onDismiss, small = true)
        }
    }
}

@Preview
@Composable
private fun MiniPlayerPreview() = PreviewSurface {
    MiniPlayer(
        NowPlaying.State("t", PreviewFixture.TITLE, PreviewFixture.ARTIST, PreviewFixture.KEY, PreviewFixture.DURATION_SEC, isPlaying = true),
        position = { 90.0 },
        onTick = {},
        onOpen = {},
        onTogglePlay = {},
        onDismiss = {},
    )
}
