package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Cloud
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.LibraryMusic
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material.icons.rounded.PlayCircle
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import com.musicdecoded.mobile.ui.theme.Tilt

/**
 * Tab page header: eyebrow + uppercase display title (e.g. "YOUR COLLECTION" / "LIBRARY").
 * [color] is the title color (`text` on ink/violet, `onColor` on a lime block); [eyebrowColor] is
 * lime on ink, white on violet, ink on lime.
 */
@Composable
fun PageHeader(
    title: String,
    modifier: Modifier = Modifier,
    overline: String? = null,
    color: Color = MusicDecodedColors.Text,
    eyebrowColor: Color = MusicDecodedColors.Lime,
) {
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        if (overline != null) Eyebrow(overline, color = eyebrowColor)
        Text(title.uppercase(), style = MaterialTheme.typography.headlineMedium, color = color, modifier = Modifier.semantics { heading() })
    }
}

/**
 * Section title (uppercase display) with an optional eyebrow, a decorative big numeral behind it
 * ("01") and a trailing action ("See all").
 */
@Composable
fun SectionHeader(
    title: String,
    modifier: Modifier = Modifier,
    overline: String? = null,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
    numeral: String? = null,
    color: Color = MusicDecodedColors.Text,
) {
    Box(modifier.fillMaxWidth()) {
        if (numeral != null) {
            BigNumeral(numeral, Modifier.align(Alignment.CenterEnd), color = color, style = MaterialTheme.typography.displayLarge.copy(fontSize = 72.sp))
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Bottom) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                if (overline != null) Eyebrow(overline)
                Text(title.uppercase(), style = MaterialTheme.typography.titleMedium, color = color, modifier = Modifier.semantics { heading() })
            }
            if (actionLabel != null && onAction != null) {
                TextButton(onClick = onAction, shape = Shapes.button) {
                    Text(actionLabel.uppercase(), style = TextStyles.sticker, color = MusicDecodedColors.Primary)
                }
            }
        }
    }
}

/**
 * Song art: the thumbnail when there is one (YouTube), else the title's initial on `surfaceRaised`
 * (the same art in song rows, the mini player and Processing), square with a white-20% 3pt border.
 * Decorative for screen readers.
 */
@Composable
fun SongArt(
    title: String,
    modifier: Modifier = Modifier,
    thumbnailUrl: String? = null,
    shape: Shape = Shapes.art,
    style: TextStyle = MaterialTheme.typography.titleMedium,
    border: Color = MusicDecodedColors.BorderOnInk,
) {
    Box(
        modifier.clip(shape).background(MusicDecodedColors.SurfaceRaised).border(Borders.thick, border, shape),
        contentAlignment = Alignment.Center,
    ) {
        if (thumbnailUrl != null) {
            RemoteImage(thumbnailUrl, Modifier.matchParentSize().padding(Borders.thick))
        } else {
            Text(titleInitial(title), style = style, color = MusicDecodedColors.Primary, modifier = Modifier.clearAndSetSemantics {})
        }
    }
}

/** The first letter or digit of [title], uppercased ("" when there's none). */
fun titleInitial(title: String): String = title.firstOrNull { it.isLetterOrDigit() }?.uppercase() ?: ""

/** Row state shown by [SongRow] (mirrors the local LibraryItem status). */
enum class SongRowStatus { ANALYZING, READY, FAILED }

/**
 * A history row: art · title · artist · meta line (`duration · key · BPM`) · offline badge · ⋯.
 * While analyzing it shows [progress] (0–100, or indeterminate when null); when failed, a retry.
 * [numeral] ("01") is drawn big and faint behind the row (decorative).
 */
@Composable
fun SongRow(
    title: String,
    artist: String?,
    meta: String,
    onClick: () -> Unit,
    onMore: () -> Unit,
    modifier: Modifier = Modifier,
    status: SongRowStatus = SongRowStatus.READY,
    progress: Int? = null,
    offline: Boolean = false,
    onRetry: (() -> Unit)? = null,
    thumbnailUrl: String? = null,
    numeral: String? = null,
) {
    Box(
        modifier
            .fillMaxWidth()
            .block(
                MusicDecodedColors.Surface,
                border = null,
                onClick = onClick,
                onClickLabel = when (status) {
                    SongRowStatus.READY -> "Open player"
                    SongRowStatus.ANALYZING -> "View progress"
                    SongRowStatus.FAILED -> null
                },
            ),
    ) {
        if (numeral != null) {
            BigNumeral(
                numeral,
                Modifier.align(Alignment.CenterEnd).padding(end = Spacing.xxxl),
                style = MaterialTheme.typography.displayLarge.copy(fontSize = 64.sp),
            )
        }
        Row(
            Modifier.fillMaxWidth().padding(Spacing.xs),
            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            SongArt(title, Modifier.size(Sizes.songArt), thumbnailUrl)
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                Text(title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                if (artist != null) {
                    Text(artist, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                when (status) {
                    SongRowStatus.READY -> Row(
                        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        if (offline) OfflineBadge()
                        Text(meta, style = TextStyles.counter, color = MusicDecodedColors.TextMuted)
                    }
                    SongRowStatus.ANALYZING -> Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                        Text(
                            if (progress != null) "Analyzing · $progress%" else "Analyzing…",
                            style = MaterialTheme.typography.labelSmall,
                            color = MusicDecodedColors.Primary,
                        )
                        if (progress != null) WrappedProgressBar(progress / 100f) else WrappedProgressBar(null)
                    }
                    SongRowStatus.FAILED -> Row(
                        horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons.Rounded.ErrorOutline, null, tint = MusicDecodedColors.Record, modifier = Modifier.size(Sizes.smallIcon))
                        Text("Failed", style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.Record)
                        if (onRetry != null) {
                            TextButton(onClick = onRetry, shape = Shapes.button) {
                                Text("Retry", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Primary)
                            }
                        }
                    }
                }
            }
            MdIconButton(Icons.Rounded.MoreHoriz, contentDescription = "More options for $title", onClick = onMore, small = true)
        }
    }
}

/**
 * Square-ended progress bar: lime on `surfaceRaised` (a null [progress] is indeterminate).
 * [height] defaults to the 4dp track.
 */
@Composable
fun WrappedProgressBar(
    progress: Float?,
    modifier: Modifier = Modifier,
    color: Color = MusicDecodedColors.Primary,
    trackColor: Color = MusicDecodedColors.SurfaceRaised,
    height: androidx.compose.ui.unit.Dp = Sizes.progressTrack,
) {
    val m = modifier.fillMaxWidth().height(height)
    if (progress != null) {
        LinearProgressIndicator(
            progress = { progress },
            modifier = m,
            color = color,
            trackColor = trackColor,
            strokeCap = StrokeCap.Butt,
            gapSize = 0.dp,
            drawStopIndicator = {},
        )
    } else {
        LinearProgressIndicator(m, color = color, trackColor = trackColor, strokeCap = StrokeCap.Butt, gapSize = 0.dp)
    }
}

/** Offline badge: a small yellow sticker with a download icon. */
@Composable
private fun OfflineBadge() {
    Box(
        Modifier.tilt(Tilt.sticker).block(MusicDecodedColors.Yellow, shadow = Shadows.sm).padding(Spacing.xxs / 2),
        contentAlignment = Alignment.Center,
    ) {
        Icon(Icons.Rounded.Download, contentDescription = "Available offline", tint = MusicDecodedColors.OnColor, modifier = Modifier.size(Sizes.smallIcon))
    }
}

/** Source fill per DESIGN.md §1 "Source tiles" (the icon is always ink on the fill). */
enum class SourceKind(val icon: ImageVector, val fill: Color) {
    FILES(Icons.Rounded.Folder, MusicDecodedColors.Lime),
    MEDIA_LIBRARY(Icons.Rounded.LibraryMusic, MusicDecodedColors.Blue),
    YOUTUBE(Icons.Rounded.PlayCircle, MusicDecodedColors.Orange),
    SOUNDCLOUD(Icons.Rounded.Cloud, MusicDecodedColors.Yellow),
    RECORD(Icons.Rounded.Mic, MusicDecodedColors.Pink),
}

/** The source's icon tile: ink icon on the source fill with a black border. */
@Composable
fun SourceIconTile(kind: SourceKind, modifier: Modifier = Modifier, fill: Color = kind.fill) {
    Box(
        modifier.size(Sizes.sourceTile).background(fill).border(Borders.thick, MusicDecodedColors.Border),
        contentAlignment = Alignment.Center,
    ) {
        Icon(kind.icon, contentDescription = null, tint = MusicDecodedColors.OnColor)
    }
}

/**
 * Home "Transcribe Music" card: a color block in the source's fill (ink text, black border, accent
 * hard shadow on ink), optionally tilted like a sticker ([tilt]).
 */
@Composable
fun SourceCard(
    kind: SourceKind,
    title: String,
    description: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    tilt: Float = 0f,
) {
    Column(
        modifier
            .tilt(tilt)
            .block(kind.fill, shadow = Shadows.md, shadowColor = MusicDecodedColors.Violet, onClick = onClick)
            .defaultMinSize(minHeight = Sizes.minTouchTarget)
            .padding(Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        SourceIconTile(kind, fill = MusicDecodedColors.White)
        Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(title.uppercase(), style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.OnColor)
            Text(description, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.OnColor)
        }
    }
}

/** Small info note, e.g. the R-1 offline-audio policy copy: a card with a blue icon. */
@Composable
fun PolicyNote(text: String, modifier: Modifier = Modifier, icon: ImageVector = Icons.Rounded.Info) {
    Row(
        modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface)
            .border(Borders.thick, MusicDecodedColors.BorderOnInk)
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Icon(icon, contentDescription = null, tint = MusicDecodedColors.Info, modifier = Modifier.size(Sizes.smallIcon))
        Text(text, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
    }
}

@Preview
@Composable
private fun SectionHeaderPreview() = PreviewSurface {
    SectionHeader("Recently Decoded", actionLabel = "See all", onAction = {}, numeral = "01")
}

@Preview
@Composable
private fun SongRowPreview() = PreviewSurface {
    val meta = songMetaLine(PreviewFixture.DURATION_SEC, PreviewFixture.KEY, PreviewFixture.BPM)
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        SongRow(PreviewFixture.TITLE, PreviewFixture.ARTIST, meta, onClick = {}, onMore = {}, offline = true, numeral = "01")
        SongRow(PreviewFixture.TITLE, PreviewFixture.ARTIST, meta, onClick = {}, onMore = {}, status = SongRowStatus.ANALYZING, progress = 42)
        SongRow(PreviewFixture.TITLE, null, meta, onClick = {}, onMore = {}, status = SongRowStatus.FAILED, onRetry = {})
    }
}

@Preview
@Composable
private fun SourceCardPreview() = PreviewSurface {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.md)) {
            SourceCard(SourceKind.FILES, "Files", "Import audio from this device", onClick = {}, Modifier.weight(1f), tilt = -Tilt.listCard)
            SourceCard(SourceKind.MEDIA_LIBRARY, "Media Library", "Choose local music", onClick = {}, Modifier.weight(1f), tilt = Tilt.listCard)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.md)) {
            SourceCard(SourceKind.YOUTUBE, "YouTube", "Discover & play", onClick = {}, Modifier.weight(1f), tilt = Tilt.listCard)
            SourceCard(SourceKind.SOUNDCLOUD, "SoundCloud", "Find supported tracks", onClick = {}, Modifier.weight(1f), tilt = -Tilt.listCard)
        }
    }
}

@Preview
@Composable
private fun PolicyNotePreview() = PreviewSurface {
    PolicyNote("Offline audio is saved only when permitted. The chord timeline can remain available separately.")
}
