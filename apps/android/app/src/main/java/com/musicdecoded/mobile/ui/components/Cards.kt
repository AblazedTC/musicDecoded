package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.LibraryMusic
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material.icons.rounded.PlayCircle
import androidx.compose.material.icons.rounded.Cloud
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles

/** Tab page header: overline + page title (e.g. "YOUR COLLECTION" / "Library"). */
@Composable
fun PageHeader(title: String, modifier: Modifier = Modifier, overline: String? = null) {
    Column(modifier.fillMaxWidth()) {
        if (overline != null) Text(overline.uppercase(), style = TextStyles.overline)
        Text(title, style = MaterialTheme.typography.headlineMedium, color = MusicDecodedColors.Text, modifier = Modifier.semantics { heading() })
    }
}

/** Section title with an optional overline and trailing action ("See all"). */
@Composable
fun SectionHeader(
    title: String,
    modifier: Modifier = Modifier,
    overline: String? = null,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
) {
    Row(modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            if (overline != null) Text(overline.uppercase(), style = TextStyles.overline)
            Text(title, style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Text, modifier = Modifier.semantics { heading() })
        }
        if (actionLabel != null && onAction != null) {
            TextButton(onClick = onAction) {
                Text(actionLabel, style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Primary)
            }
        }
    }
}

/**
 * Song art: the thumbnail when there is one (YouTube), else the title's initial on `surfaceSoft`
 * (the same art in song rows, the mini player and Processing). Decorative for screen readers.
 */
@Composable
fun SongArt(
    title: String,
    modifier: Modifier = Modifier,
    thumbnailUrl: String? = null,
    shape: Shape = Shapes.art,
    style: TextStyle = MaterialTheme.typography.titleMedium,
) {
    Box(modifier.clip(shape).background(MusicDecodedColors.SurfaceSoft), contentAlignment = Alignment.Center) {
        if (thumbnailUrl != null) {
            RemoteImage(thumbnailUrl, Modifier.matchParentSize())
        } else {
            Text(titleInitial(title), style = style, color = MusicDecodedColors.Secondary, modifier = Modifier.clearAndSetSemantics {})
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
) {
    Row(
        modifier
            .fillMaxWidth()
            .clip(Shapes.row)
            .background(MusicDecodedColors.Surface)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.row)
            .clickable(
                role = Role.Button,
                onClickLabel = when (status) {
                    SongRowStatus.READY -> "Open player"
                    SongRowStatus.ANALYZING -> "View progress"
                    SongRowStatus.FAILED -> null
                },
                onClick = onClick,
            )
            .padding(Spacing.xs),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        SongArt(title, Modifier.size(Sizes.songArt), thumbnailUrl)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(
                title,
                style = MaterialTheme.typography.titleSmall,
                color = MusicDecodedColors.Text,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            if (artist != null) {
                Text(artist, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            when (status) {
                SongRowStatus.READY -> Row(
                    horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (offline) OfflineBadge()
                    Text(meta, style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextMuted)
                }
                SongRowStatus.ANALYZING -> Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                    Text(
                        if (progress != null) "Analyzing · $progress%" else "Analyzing…",
                        style = MaterialTheme.typography.labelSmall,
                        color = MusicDecodedColors.Primary,
                    )
                    val trackModifier = Modifier.fillMaxWidth().height(Sizes.progressTrack).clip(CircleShape)
                    if (progress != null) {
                        LinearProgressIndicator(
                            progress = { progress / 100f },
                            modifier = trackModifier,
                            color = MusicDecodedColors.Primary,
                            trackColor = MusicDecodedColors.SurfaceSoft,
                        )
                    } else {
                        LinearProgressIndicator(trackModifier, color = MusicDecodedColors.Primary, trackColor = MusicDecodedColors.SurfaceSoft)
                    }
                }
                SongRowStatus.FAILED -> Row(
                    horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons.Rounded.ErrorOutline, null, tint = MusicDecodedColors.Record, modifier = Modifier.size(Sizes.smallIcon))
                    Text("Failed", style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.Record)
                    if (onRetry != null) {
                        TextButton(onClick = onRetry) {
                            Text("Retry", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Primary)
                        }
                    }
                }
            }
        }
        MdIconButton(Icons.Rounded.MoreHoriz, contentDescription = "More options for $title", onClick = onMore, small = true)
    }
}

@Composable
private fun OfflineBadge() {
    Box(
        Modifier.background(MusicDecodedColors.PrimarySoft, Shapes.chip).padding(Spacing.xxs),
        contentAlignment = Alignment.Center,
    ) {
        Icon(Icons.Rounded.Download, contentDescription = "Available offline", tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
    }
}

/** Icon tint pair per source (DESIGN.md §1 "Source-icon tints"). */
enum class SourceKind(val icon: ImageVector, val tint: Color, val background: Color) {
    FILES(Icons.Rounded.Folder, MusicDecodedColors.Primary, MusicDecodedColors.PrimarySoft),
    MEDIA_LIBRARY(Icons.Rounded.LibraryMusic, MusicDecodedColors.Secondary, MusicDecodedColors.MediaLibraryBg),
    YOUTUBE(Icons.Rounded.PlayCircle, MusicDecodedColors.YouTube, MusicDecodedColors.YouTubeBg),
    SOUNDCLOUD(Icons.Rounded.Cloud, MusicDecodedColors.SoundCloud, MusicDecodedColors.SoundCloudBg),
    RECORD(Icons.Rounded.Mic, MusicDecodedColors.RecordSource, MusicDecodedColors.RecordSourceBg),
}

/** Home "Transcribe Music" card: icon tile, title, description. Gradient from `surfaceRaised`. */
@Composable
fun SourceCard(
    kind: SourceKind,
    title: String,
    description: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier
            .clip(Shapes.card)
            .background(Brush.verticalGradient(listOf(MusicDecodedColors.SurfaceRaised, MusicDecodedColors.Surface)))
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .clickable(role = Role.Button, onClick = onClick)
            .defaultMinSize(minHeight = Sizes.minTouchTarget)
            .padding(Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Box(Modifier.size(Sizes.sourceTile).background(kind.background, Shapes.control), contentAlignment = Alignment.Center) {
            Icon(kind.icon, contentDescription = null, tint = kind.tint)
        }
        Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            Text(description, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
    }
}

/** Small info note, e.g. the R-1 offline-audio policy copy. */
@Composable
fun PolicyNote(text: String, modifier: Modifier = Modifier, icon: ImageVector = Icons.Rounded.Info) {
    Row(
        modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.control)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.control)
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Icon(icon, contentDescription = null, tint = MusicDecodedColors.Secondary, modifier = Modifier.size(Sizes.smallIcon))
        Text(text, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
    }
}

@Preview
@Composable
private fun SectionHeaderPreview() = PreviewSurface {
    SectionHeader("Recently Decoded", actionLabel = "See all", onAction = {})
}

@Preview
@Composable
private fun SongRowPreview() = PreviewSurface {
    val meta = songMetaLine(PreviewFixture.DURATION_SEC, PreviewFixture.KEY, PreviewFixture.BPM)
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        SongRow(PreviewFixture.TITLE, PreviewFixture.ARTIST, meta, onClick = {}, onMore = {}, offline = true)
        SongRow(PreviewFixture.TITLE, PreviewFixture.ARTIST, meta, onClick = {}, onMore = {}, status = SongRowStatus.ANALYZING, progress = 42)
        SongRow(PreviewFixture.TITLE, null, meta, onClick = {}, onMore = {}, status = SongRowStatus.FAILED, onRetry = {})
    }
}

@Preview
@Composable
private fun SourceCardPreview() = PreviewSurface {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            SourceCard(SourceKind.FILES, "Files", "Import audio from this device", onClick = {}, Modifier.weight(1f))
            SourceCard(SourceKind.MEDIA_LIBRARY, "Media Library", "Choose local music", onClick = {}, Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            SourceCard(SourceKind.YOUTUBE, "YouTube", "Discover & play", onClick = {}, Modifier.weight(1f))
            SourceCard(SourceKind.SOUNDCLOUD, "SoundCloud", "Find supported tracks", onClick = {}, Modifier.weight(1f))
        }
    }
}

@Preview
@Composable
private fun PolicyNotePreview() = PreviewSurface {
    PolicyNote("Offline audio is saved only when permitted. The chord timeline can remain available separately.")
}
