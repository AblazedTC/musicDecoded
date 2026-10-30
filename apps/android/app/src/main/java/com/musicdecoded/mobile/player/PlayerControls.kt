package com.musicdecoded.mobile.player

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FavoriteBorder
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.tabular

/** Back · title/artist (centered) · favorite · [menu] (⋯). [onFavorite] `null` hides the heart. */
@Composable
internal fun PlayerHeader(
    title: String,
    artist: String?,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
    favorite: Boolean = false,
    onFavorite: (() -> Unit)? = null,
    menu: (@Composable () -> Unit)? = null,
) {
    Row(modifier.fillMaxWidth().padding(horizontal = Spacing.xs), verticalAlignment = Alignment.CenterVertically) {
        MdIconButton(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "Back", onClick = onBack)
        // Title on one line; the artist goes under it only if both fit the row's touch-target height.
        Layout(
            content = {
                Text(
                    title,
                    style = MaterialTheme.typography.titleSmall,
                    color = MusicDecodedColors.Text,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    textAlign = TextAlign.Center,
                )
                if (artist != null) {
                    Text(artist, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            },
            modifier = Modifier.weight(1f).semantics(mergeDescendants = true) { heading() },
        ) { measurables, constraints ->
            val loose = constraints.copy(minWidth = 0, minHeight = 0)
            val titlePlaceable = measurables[0].measure(loose)
            val artistPlaceable = measurables.getOrNull(1)?.measure(loose)
                ?.takeIf { titlePlaceable.height + it.height <= Sizes.minTouchTarget.roundToPx() }
            val width = constraints.maxWidth
            layout(width, titlePlaceable.height + (artistPlaceable?.height ?: 0)) {
                titlePlaceable.place((width - titlePlaceable.width) / 2, 0)
                artistPlaceable?.place((width - artistPlaceable.width) / 2, titlePlaceable.height)
            }
        }
        if (onFavorite != null) {
            MdIconButton(
                if (favorite) Icons.Rounded.Favorite else Icons.Rounded.FavoriteBorder,
                contentDescription = if (favorite) "Remove from favorites" else "Add to favorites",
                onClick = onFavorite,
                active = favorite,
            )
        } else {
            Box(Modifier.size(Sizes.minTouchTarget))
        }
        menu?.invoke()
    }
}

/** The pink play/pause circle with a black border and hard shadow (56dp in the transport; larger in the learning view). */
@Composable
internal fun PlayButton(isPlaying: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier, size: Dp = Sizes.playButton) {
    Box(
        modifier
            .size(size)
            .block(MusicDecodedColors.Secondary, shadow = Shadows.sm, shape = CircleShape, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            if (isPlaying) Icons.Rounded.Pause else Icons.Rounded.PlayArrow,
            contentDescription = if (isPlaying) "Pause" else "Play",
            tint = MusicDecodedColors.OnSecondary,
        )
    }
}

/** Speed: tap toggles 1.00× ⇄ 0.75×; long-press picks 0.50–1.25× in 0.05 steps (DESIGN.md §3.7). */
@Composable
internal fun SpeedButton(speed: Float, onToggle: () -> Unit, onSelect: (Float) -> Unit, modifier: Modifier = Modifier) {
    var menu by remember { mutableStateOf(false) }
    Box(modifier) {
        Box(
            Modifier
                .defaultMinSize(minWidth = Sizes.minTouchTarget, minHeight = Sizes.minTouchTarget)
                .clip(Shapes.chip)
                .combinedClickable(role = Role.Button, onClickLabel = "Toggle practice speed", onLongClickLabel = "Choose speed", onLongClick = { menu = true }, onClick = onToggle),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                formatSpeed(speed),
                style = MaterialTheme.typography.labelLarge.tabular(),
                color = if (speed != 1f) MusicDecodedColors.Primary else MusicDecodedColors.TextSecondary,
            )
        }
        ChoiceMenu(menu, onDismiss = { menu = false }, choices = PlayerViewModel.SPEED_STEPS, label = ::formatSpeed) {
            onSelect(it)
            menu = false
        }
    }
}

/** Transpose chip −6…+6 (display only). */
@Composable
internal fun TransposeChip(transpose: Int, onSelect: (Int) -> Unit, modifier: Modifier = Modifier, compact: Boolean = false) {
    var menu by remember { mutableStateOf(false) }
    Box(modifier) {
        ControlChip("Transpose", formatTranspose(transpose), onClick = { menu = true }, Modifier.fillMaxWidth(), compact = compact)
        val range = -PlayerViewModel.MAX_TRANSPOSE..PlayerViewModel.MAX_TRANSPOSE
        ChoiceMenu(menu, onDismiss = { menu = false }, choices = range.toList(), label = ::formatTranspose) {
            onSelect(it)
            menu = false
        }
    }
}

/** Speed chip for the learning view: tap opens the speed list. */
@Composable
internal fun SpeedChip(speed: Float, onSelect: (Float) -> Unit, modifier: Modifier = Modifier) {
    var menu by remember { mutableStateOf(false) }
    Box(modifier) {
        ControlChip("Speed", formatSpeed(speed), onClick = { menu = true }, Modifier.fillMaxWidth())
        ChoiceMenu(menu, onDismiss = { menu = false }, choices = PlayerViewModel.SPEED_STEPS, label = ::formatSpeed) {
            onSelect(it)
            menu = false
        }
    }
}

@Composable
private fun <T> ChoiceMenu(expanded: Boolean, onDismiss: () -> Unit, choices: List<T>, label: (T) -> String, onChoose: (T) -> Unit) {
    DropdownMenu(expanded = expanded, onDismissRequest = onDismiss, containerColor = MusicDecodedColors.SurfaceRaised) {
        choices.forEach { choice ->
            DropdownMenuItem(
                text = { Text(label(choice), style = MaterialTheme.typography.bodyMedium.tabular(), color = MusicDecodedColors.Text) },
                onClick = { onChoose(choice) },
            )
        }
    }
}

/** "Looping 1:04–1:20 · Clear", floating above the transport (pink sticker block). */
@Composable
internal fun LoopPill(loop: OpenEndRange<Double>, onClear: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier
            .block(MusicDecodedColors.Secondary, shadow = Shadows.sm)
            .padding(start = Spacing.md),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
    ) {
        Text(
            "Looping ${formatDuration(loop.start)}–${formatDuration(loop.endExclusive)} ·",
            style = MaterialTheme.typography.labelLarge.tabular(),
            color = MusicDecodedColors.OnSecondary,
        )
        Box(
            Modifier
                .defaultMinSize(minHeight = Sizes.minTouchTarget)
                .clickable(role = Role.Button, onClick = onClear)
                .padding(end = Spacing.md, start = Spacing.xxs),
            contentAlignment = Alignment.Center,
        ) {
            Text("Clear", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.OnSecondary)
        }
    }
}
