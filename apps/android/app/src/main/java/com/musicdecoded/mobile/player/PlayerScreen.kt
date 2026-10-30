package com.musicdecoded.mobile.player

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FavoriteBorder
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.OpenInFull
import androidx.compose.material.icons.rounded.Repeat
import androidx.compose.material.icons.rounded.SkipNext
import androidx.compose.material.icons.rounded.SkipPrevious
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.TextButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.ui.components.ChipRow
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.InfoChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.SegmentedToggle
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.components.keyName
import com.musicdecoded.mobile.ui.library.SongListMessages

import com.musicdecoded.mobile.ui.components.Eyebrow
import com.musicdecoded.mobile.ui.components.WaveformBars
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.tilt
import com.musicdecoded.mobile.ui.components.topRule
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Tilt
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import com.musicdecoded.mobile.ui.theme.tabular
import kotlin.math.roundToInt

/** The Player (DESIGN.md §3.7). [onExit] leaves after a ⋯ menu Delete / Re-analyze. */
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

        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = Spacing.gutter),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            MetadataRow(analysis, Modifier.padding(top = Spacing.xs))
            ChordTimeline(
                analysis = analysis,
                position = { position.value },
                chordName = chords::name,
                onSeek = viewModel::seekTo,
                loop = state.loop,
                onLoopSelected = viewModel::setLoop,
                currentChord = currentLabel?.let(chords::name)?.ifEmpty { null },
            )
            NowPlaying(currentLabel, chords, state.isPlaying)
            ChipRow {
                ControlChip(
                    "Chords",
                    if (state.chordsMode == ChordsMode.BASIC) "Basic" else "Full",
                    onClick = viewModel::toggleChordsMode,
                    Modifier.weight(1f),
                )
                ControlChip("Voicing", "Simple", onClick = {}, Modifier.weight(1f))
                TransposeChip(state.transpose, viewModel::setTranspose, Modifier.weight(1f))
            }
            VisualizerCard(
                state,
                currentLabel,
                onSelect = viewModel::selectVisualizer,
                video = {
                    track.videoId?.let { PlayerVideoPanel(viewModel, it, analysis.meta.durationSec) }
                },
                trailing = { MdIconButton(Icons.Rounded.OpenInFull, contentDescription = "Open piano view", onClick = onOpenPiano) },
                piano = {
                    PianoKeyboard(
                        notes = currentLabel?.let(chords::notes).orEmpty(),
                        bassNote = currentLabel?.let(chords::bassNote),

                        description = currentLabel?.let(chords::noteNames),
                    )
                },
            )
            Spacer(Modifier.height(Spacing.xs))
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            state.loop?.let { LoopPill(it, onClear = viewModel::clearLoop, Modifier.padding(bottom = Spacing.xs)) }
            TransportSheet(state, position, analysis.meta.durationSec, viewModel)
        }
    }
}

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

/** Key · Time signature · BPM. */
@Composable
private fun MetadataRow(analysis: Analysis, modifier: Modifier = Modifier) {
    ChipRow(modifier) {
        InfoChip(Icons.Rounded.MusicNote, "Key", analysis.key?.let(::keyName) ?: "—", Modifier.weight(1f))
        InfoChip(
            Icons.Rounded.GraphicEq,
            "Time",
            analysis.timeSignature?.let { "${it.numerator}/${it.denominator}" } ?: "—",
            Modifier.weight(1f),
        )
        InfoChip(Icons.Rounded.Speed, "Tempo", analysis.bpm?.let { "${it.roundToInt()} BPM" } ?: "—", Modifier.weight(1f))
    }
}

/**
 * The now-playing card (DESIGN.md §1, portfolio `NowPlayingCard.tsx`): `card` fill, 3pt black
 * border, pink `shadowXl`, tilted -1.5°; lime eyebrow + waveform bars, then the big lime chord.
 */
@Composable
private fun NowPlaying(label: String?, chords: ChordDisplay, isPlaying: Boolean) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(vertical = Spacing.xs)
            .tilt(Tilt.hero)
            .block(MusicDecodedColors.Surface, shadow = Shadows.xl, shadowColor = MusicDecodedColors.Secondary)
            .padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.xxs),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Eyebrow("Now playing", Modifier.weight(1f).semantics { heading() })
            if (isPlaying) WaveformBars(Modifier.size(width = Spacing.xxl, height = Spacing.md), bars = 5)
        }
        Text(label?.let(chords::name)?.ifEmpty { null } ?: "—", style = TextStyles.currentChord)
        val notes = label?.let(chords::noteNames).orEmpty()
        if (notes.isNotEmpty()) {
            Text(notes, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            Text("Representative root-position voicing", style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextMuted)
        }
    }
}

/** Piano | Video card (DESIGN.md §3.7 item 6). Video is disabled unless the source is youtube. */
@Composable
private fun VisualizerCard(
    state: PlayerUiState,
    label: String?,
    onSelect: (Visualizer) -> Unit,
    trailing: @Composable () -> Unit = {},
    piano: @Composable () -> Unit = {},
    video: @Composable () -> Unit = {},
) {
    Column(
        Modifier
            .fillMaxWidth()
            .block(MusicDecodedColors.Surface, shadow = Shadows.lg, shadowColor = MusicDecodedColors.Violet)
            .padding(Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            SegmentedToggle(
                options = listOf("Piano", "Video"),
                selectedIndex = state.visualizer.ordinal,
                onSelect = { onSelect(Visualizer.entries[it]) },
                modifier = Modifier.weight(1f),
                enabled = { it == Visualizer.PIANO.ordinal || state.videoAvailable },
            )
            trailing()
        }
        // When the video is the clock it stays on screen under Piano too (no hidden playback);
        // one call site keeps the same WebView across the toggle.
        if (state.visualizer == Visualizer.VIDEO || state.videoIsClock) video()
        when (state.visualizer) {
            Visualizer.PIANO -> {
                Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                    Eyebrow("Current chord")
                    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs), verticalAlignment = Alignment.Bottom) {
                        Text(
                            label?.let(state.chords::name)?.ifEmpty { null } ?: "—",
                            style = MaterialTheme.typography.titleMedium,
                            color = MusicDecodedColors.Chord,
                        )
                        Text(label?.let(state.chords::noteNames).orEmpty(), style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
                    }
                }
                piano()
            }
            Visualizer.VIDEO -> Unit
        }
    }
}

/** Seek bar + speed · previous chord · play/pause · next chord · loop (DESIGN.md §3.7 item 7). */
@Composable
private fun TransportSheet(state: PlayerUiState, position: State<Double>, duration: Double, viewModel: PlayerViewModel) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface)
            .topRule()
            .padding(horizontal = Spacing.md, vertical = Spacing.sm),
    ) {
        SeekBar(position, duration, viewModel::seekTo)
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
 * a small square lime thumb, plus current / total times. The touch target keeps the Slider's 48dp height.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SeekBar(position: State<Double>, duration: Double, onSeek: (Double) -> Unit) {
    val total = duration.toFloat().coerceAtLeast(1f)
    val value = position.value.toFloat().coerceIn(0f, total)
    Column {
        Slider(
            value = value,
            onValueChange = { onSeek(it.toDouble()) },
            valueRange = 0f..total,
            modifier = Modifier.semantics {
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
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            val style = MaterialTheme.typography.labelSmall.tabular()
            Text(formatDuration(position.value), style = style, color = MusicDecodedColors.TextSecondary)
            Text(formatDuration(duration), style = style, color = MusicDecodedColors.TextMuted)
        }
    }
}

/** The seek bar's square thumb (a component size, not spacing). */
private val SeekThumb = 14.dp

@Preview(heightDp = 1100)
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
