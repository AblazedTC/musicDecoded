package com.musicdecoded.mobile.player

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FavoriteBorder
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.OpenInFull
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Repeat
import androidx.compose.material.icons.rounded.SkipNext
import androidx.compose.material.icons.rounded.SkipPrevious
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.MetaChip
import com.musicdecoded.mobile.ui.components.SegmentedToggle
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.components.keyName
import com.musicdecoded.mobile.ui.components.topRule
import com.musicdecoded.mobile.ui.library.SongListMessages
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.tabular
import kotlin.math.roundToInt

/**
 * The Player (DESIGN.md §3.7), one screen with no vertical scrolling: header · metadata chips ·
 * edge-to-edge timeline (~32% of the middle space) · control row · visualizer panel (the rest) ·
 * transport at the bottom. At font scale ≥ [SCROLL_FONT_SCALE] it scrolls as a fallback.
 * [onExit] leaves after a ⋯ menu Delete / Re-analyze.
 */
@Composable
fun PlayerScreen(viewModel: PlayerViewModel, onBack: () -> Unit, onOpenPiano: () -> Unit, onExit: (PlayerExit) -> Unit = { onBack() }) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val position = viewModel.position.collectAsStateWithLifecycle()
    PlaybackClock(viewModel, state.isPlaying)
    SongListMessages(state.message, viewModel::messageShown)
    LaunchedEffect(state.exit) {
        state.exit?.let {
            viewModel.exitHandled()
            onExit(it)
        }
    }

    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        val track = state.track
        PlayerHeader(
            title = track?.title.orEmpty(),
            artist = track?.artist,
            onBack = onBack,
            favorite = state.favorite,
            onFavorite = viewModel::toggleFavorite.takeIf { track != null },
            menu = if (track != null) {
                { PlayerMenu(track.title, state.favorite, viewModel::toggleFavorite, viewModel::reanalyze, viewModel::delete) }
            } else null,
        )
        if (track == null) {
            PlayerPlaceholder(state.loading, Modifier.weight(1f))
            return@Column
        }
        val analysis = track.analysis
        val chords = state.chords
        val chordIndex by remember(analysis) { derivedStateOf { currentState(position.value, analysis).chordIndex } }
        val currentLabel = chordIndex?.let { analysis.chords[it].label }
        val scrolls = LocalDensity.current.fontScale >= SCROLL_FONT_SCALE

        Column(Modifier.weight(1f).then(if (scrolls) Modifier.verticalScroll(rememberScrollState()) else Modifier)) {
            MetadataRow(analysis, Modifier.padding(horizontal = Spacing.gutter, vertical = Spacing.xxs))
            BoxWithConstraints(if (scrolls) Modifier else Modifier.weight(1f)) {
                val timelineHeight = if (scrolls) TIMELINE_MAX else (maxHeight * TIMELINE_SHARE).coerceIn(TIMELINE_MIN, TIMELINE_MAX)
                Column(if (scrolls) Modifier else Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                    ChordTimeline(
                        analysis = analysis,
                        position = { position.value },
                        chords = chords,
                        currentIndex = chordIndex,
                        onSeek = viewModel::seekTo,
                        modifier = Modifier.height(timelineHeight),
                        loop = state.loop,
                        onLoopSelected = viewModel::setLoop,
                    )
                    Row(Modifier.padding(horizontal = Spacing.gutter), horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                        ControlChip(
                            "Chords",
                            if (state.chordsMode == ChordsMode.BASIC) "Basic" else "Full",
                            onClick = viewModel::toggleChordsMode,
                            Modifier.weight(1f),
                            compact = true,
                        )
                        ControlChip("Voicing", "Simple", onClick = {}, Modifier.weight(1f), compact = true)
                        TransposeChip(state.transpose, viewModel::setTranspose, Modifier.weight(1f), compact = true)
                    }
                    VisualizerPanel(
                        state,
                        currentLabel,
                        onSelect = viewModel::selectVisualizer,
                        modifier = if (scrolls) Modifier.height(SCROLL_PANEL_HEIGHT) else Modifier.weight(1f),
                        video = { modifier ->
                            track.videoId?.let { PlayerVideoPanel(viewModel, it, analysis.meta.durationSec, modifier) }
                        },
                        trailing = { MdIconButton(Icons.Rounded.OpenInFull, contentDescription = "Open piano view", onClick = onOpenPiano) },
                    )
                }
            }
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            state.loop?.let { LoopPill(it, onClear = viewModel::clearLoop, Modifier.padding(vertical = Spacing.xxs)) }
            TransportSheet(state, position, analysis.meta.durationSec, viewModel)
        }
    }
}

/** Timeline height: this share of the space between the metadata row and the transport, clamped (DESIGN.md §3.7). */
private const val TIMELINE_SHARE = 0.32f
private val TIMELINE_MIN = 150.dp
private val TIMELINE_MAX = 220.dp

/** At this font scale the player may scroll (DESIGN.md §3.7 accessibility); the panel then gets a fixed height. */
private const val SCROLL_FONT_SCALE = 1.5f
private val SCROLL_PANEL_HEIGHT = 280.dp

/** Header ⋯: Favorite · Re-analyze · Delete (with confirmation), like the song rows. */
@Composable
private fun PlayerMenu(title: String, favorite: Boolean, onFavorite: () -> Unit, onReanalyze: () -> Unit, onDelete: () -> Unit) {
    var open by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    Box {
        MdIconButton(Icons.Rounded.MoreHoriz, contentDescription = "More options", onClick = { open = true })
        DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = MusicDecodedColors.SurfaceRaised) {
            DropdownMenuItem(
                text = { Text(if (favorite) "Remove from favorites" else "Favorite") },
                leadingIcon = { Icon(if (favorite) Icons.Rounded.Favorite else Icons.Rounded.FavoriteBorder, null) },
                onClick = {
                    open = false
                    onFavorite()
                },
            )
            DropdownMenuItem(
                text = { Text("Re-analyze") },
                leadingIcon = { Icon(Icons.Rounded.Refresh, null) },
                onClick = {
                    open = false
                    onReanalyze()
                },
            )
            DropdownMenuItem(
                text = { Text("Delete", color = MusicDecodedColors.Record) },
                leadingIcon = { Icon(Icons.Rounded.Delete, null, tint = MusicDecodedColors.Record) },
                onClick = {
                    open = false
                    confirmDelete = true
                },
            )
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            containerColor = MusicDecodedColors.Surface,
            title = { Text("Delete “$title”?") },
            text = {
                Text(
                    "It's removed from your library and its chord timeline won't show up here anymore.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.TextSecondary,
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    onDelete()
                }) { Text("Delete", color = MusicDecodedColors.Record) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete = false }) { Text("Cancel", color = MusicDecodedColors.TextSecondary) }
            },
        )
    }
}

@Composable
private fun PlayerPlaceholder(loading: Boolean, modifier: Modifier = Modifier) {
    Box(modifier.fillMaxSize().padding(Spacing.gutter), contentAlignment = Alignment.Center) {
        if (loading) {
            CircularProgressIndicator(color = MusicDecodedColors.Primary)
        } else {
            Text(
                "This transcription isn't on this device.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
        }
    }
}

/** Key · Time signature · BPM as compact chips with the value inline (room for more; scrolls sideways if needed). */
@Composable
private fun MetadataRow(analysis: Analysis, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        MetaChip(Icons.Rounded.MusicNote, "Key", analysis.key?.let(::keyName) ?: "—")
        MetaChip(Icons.Rounded.GraphicEq, "Time", analysis.timeSignature?.let { "${it.numerator}/${it.denominator}" } ?: "—")
        MetaChip(Icons.Rounded.Speed, "Tempo", analysis.bpm?.let { "${it.roundToInt()} BPM" } ?: "—")
    }
}

/**
 * Visualizer panel (DESIGN.md §3.7 item 5), filling the height it's given: compact Piano | Video
 * tabs + expand, then the active chord (name + numeral) over a keyboard that fills the rest, or the
 * letterboxed video. Video is disabled unless the source is youtube.
 */
@Composable
private fun VisualizerPanel(
    state: PlayerUiState,
    label: String?,
    onSelect: (Visualizer) -> Unit,
    modifier: Modifier = Modifier,
    trailing: @Composable () -> Unit = {},
    video: @Composable (Modifier) -> Unit = {},
) {
    Column(
        modifier
            .fillMaxWidth()
            .padding(start = Spacing.gutter, end = Spacing.gutter, bottom = Spacing.xs)
            .block(MusicDecodedColors.Surface, shadow = Shadows.lg, shadowColor = MusicDecodedColors.Violet)
            .padding(Spacing.xs),
        verticalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            SegmentedToggle(
                options = listOf("Piano", "Video"),
                selectedIndex = state.visualizer.ordinal,
                onSelect = { onSelect(Visualizer.entries[it]) },
                modifier = Modifier.weight(1f),
                enabled = { it == Visualizer.PIANO.ordinal || state.videoAvailable },
                segmentHeight = Sizes.compactTab - Spacing.xs,
            )
            trailing()
        }
        // When the video is the clock it stays on screen under Piano too (no hidden playback);
        // one call site keeps the same WebView across the toggle.
        val showVideo = state.visualizer == Visualizer.VIDEO || state.videoIsClock
        if (showVideo) {
            Box(Modifier.weight(1f).fillMaxWidth().clipToBounds(), contentAlignment = Alignment.Center) { video(Modifier) }
        }
        if (state.visualizer == Visualizer.PIANO) {
            val chords = state.chords
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(Spacing.xs, Alignment.CenterHorizontally),
                verticalAlignment = Alignment.Bottom,
            ) {
                Text(
                    label?.let(chords::name)?.ifEmpty { null } ?: "—",
                    style = MaterialTheme.typography.titleMedium,
                    color = MusicDecodedColors.Chord,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.alignByBaseline(),
                )
                label?.let(chords::numeral)?.let {
                    Text(it, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.TextSecondary, modifier = Modifier.alignByBaseline())
                }
            }
            FittedPianoKeyboard(
                notes = label?.let(chords::notes).orEmpty(),
                noteName = chords::noteName,
                bassNote = label?.let(chords::bassNote),
                description = label?.let(chords::noteNames),
                modifier = Modifier.weight(if (showVideo) 1.5f else 1f).fillMaxWidth(),
            )
        }
    }
}

/** Seek bar with times on one line, then speed · previous chord · play/pause · next chord · loop (DESIGN.md §3.7 item 6). */
@Composable
private fun TransportSheet(state: PlayerUiState, position: State<Double>, duration: Double, viewModel: PlayerViewModel) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface)
            .topRule()
            .padding(horizontal = Spacing.md, vertical = Spacing.xxs),
    ) {
        SeekBar(position, duration, viewModel::seekTo, inlineTimes = true)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            SpeedButton(state.speed, onToggle = viewModel::toggleSpeed, onSelect = viewModel::setSpeed)
            MdIconButton(Icons.Rounded.SkipPrevious, contentDescription = "Previous chord", onClick = viewModel::previousChord)
            PlayButton(state.isPlaying, onClick = viewModel::togglePlay)
            MdIconButton(Icons.Rounded.SkipNext, contentDescription = "Next chord", onClick = viewModel::nextChord)
            MdIconButton(
                Icons.Rounded.Repeat,
                contentDescription = if (state.loop != null) "Stop looping" else "Loop current bar",
                onClick = viewModel::toggleLoop,
                active = state.loop != null,
            )
        }
    }
}

/**
 * Seek bar per DESIGN.md §3.7: a thin square track (played = `primary`, rest = `surfaceRaised`) with
 * a small square lime thumb, plus current / total times (under the bar, or either side of it with
 * [inlineTimes]). The touch target keeps the Slider's 48dp height.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SeekBar(position: State<Double>, duration: Double, onSeek: (Double) -> Unit, inlineTimes: Boolean = false) {
    val total = duration.toFloat().coerceAtLeast(1f)
    val value = position.value.toFloat().coerceIn(0f, total)
    val timeStyle = MaterialTheme.typography.labelSmall.tabular()
    val slider = @Composable { modifier: Modifier ->
        Slider(
            value = value,
            onValueChange = { onSeek(it.toDouble()) },
            valueRange = 0f..total,
            modifier = modifier.semantics {
                contentDescription = "Seek"
                stateDescription = "${formatDuration(position.value)} of ${formatDuration(duration)}"
            },
            thumb = { Box(Modifier.size(SeekThumb).background(MusicDecodedColors.Primary).border(Borders.separator, MusicDecodedColors.Border)) },
            track = {
                Canvas(Modifier.fillMaxWidth().height(Sizes.progressTrack)) {
                    drawRect(MusicDecodedColors.SurfaceRaised)
                    drawRect(MusicDecodedColors.Primary, size = Size(size.width * (value / total), size.height))
                }
            },
        )
    }
    if (inlineTimes) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            Text(formatDuration(position.value), style = timeStyle, color = MusicDecodedColors.TextSecondary)
            slider(Modifier.weight(1f))
            Text(formatDuration(duration), style = timeStyle, color = MusicDecodedColors.TextMuted)
        }
    } else {
        Column {
            slider(Modifier)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(formatDuration(position.value), style = timeStyle, color = MusicDecodedColors.TextSecondary)
                Text(formatDuration(duration), style = timeStyle, color = MusicDecodedColors.TextMuted)
            }
        }
    }
}

/** The seek bar's square thumb (a component size, not spacing). */
private val SeekThumb = 14.dp

@Preview(widthDp = 412, heightDp = 860)
@Composable
private fun PlayerScreenPreview() = MusicDecodedTheme {
    val analysis = rememberPreviewAnalysis() ?: return@MusicDecodedTheme
    val context = LocalContext.current
    val viewModel = remember {
        PlayerViewModel(
            loadTrack = { PlayerTrack(analysis, "Sunny Days", "Fixture", favorite = true, audio = null) },
            voicings = runCatching { loadChordVoicings(context) }.getOrNull(),
            createEngine = { SilentPlaybackEngine(it.analysis.meta.durationSec) },
        ).apply { seekTo(16.0) }
    }
    PlayerScreen(viewModel, onBack = {}, onOpenPiano = {})
}
