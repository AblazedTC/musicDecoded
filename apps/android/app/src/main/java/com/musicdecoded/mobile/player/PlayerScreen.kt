package com.musicdecoded.mobile.player

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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.OpenInFull
import androidx.compose.material.icons.rounded.Repeat
import androidx.compose.material.icons.rounded.SkipNext
import androidx.compose.material.icons.rounded.SkipPrevious
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.InfoChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.SegmentedToggle
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.components.keyName
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import com.musicdecoded.mobile.ui.theme.tabular
import kotlin.math.roundToInt

/** The Player (DESIGN.md §3.7). */
@Composable
fun PlayerScreen(viewModel: PlayerViewModel, onBack: () -> Unit, onOpenPiano: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val position = viewModel.position.collectAsStateWithLifecycle()
    PlaybackClock(viewModel, state.isPlaying)

    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        val track = state.track
        PlayerHeader(
            title = track?.title.orEmpty(),
            artist = track?.artist,
            onBack = onBack,
            favorite = state.favorite,
            onFavorite = viewModel::toggleFavorite.takeIf { track != null },
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
            MetadataRow(analysis)
            ChordTimeline(
                analysis = analysis,
                position = { position.value },
                chordName = chords::name,
                onSeek = viewModel::seekTo,
                loop = state.loop,
                onLoopSelected = viewModel::setLoop,
            )
            NowPlaying(currentLabel, chords)
            Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
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
                trailing = { MdIconButton(Icons.Rounded.OpenInFull, contentDescription = "Open piano view", onClick = onOpenPiano) },
                piano = {
                    PianoKeyboard(
                        notes = currentLabel?.let(chords::notes).orEmpty(),
                        bassNote = currentLabel?.let(chords::bassNote),
                        modifier = Modifier.clip(Shapes.chip),
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
private fun MetadataRow(analysis: Analysis) {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
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

@Composable
private fun NowPlaying(label: String?, chords: ChordDisplay) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
        Text("NOW PLAYING", style = TextStyles.overline)
        Text(label?.let(chords::name)?.ifEmpty { null } ?: "—", style = TextStyles.currentChord)
        val notes = label?.let(chords::noteNames).orEmpty()
        if (notes.isNotEmpty()) {
            Text(notes, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.TextSecondary)
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
) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
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
        when (state.visualizer) {
            Visualizer.PIANO -> {
                Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                    Text("CURRENT CHORD", style = TextStyles.overline)
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
            // The synced YouTube embed arrives with T-305a.
            Visualizer.VIDEO -> Text(
                "Video playback is coming soon.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
        }
    }
}

/** Seek bar + speed · previous chord · play/pause · next chord · loop (DESIGN.md §3.7 item 7). */
@Composable
private fun TransportSheet(state: PlayerUiState, position: State<Double>, duration: Double, viewModel: PlayerViewModel) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.TransportBg, Shapes.sheet)
            .border(Sizes.hairline, MusicDecodedColors.TransportBorder, Shapes.sheet)
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

/** Seek slider with current / total times. */
@Composable
internal fun SeekBar(position: State<Double>, duration: Double, onSeek: (Double) -> Unit) {
    Column {
        Slider(
            value = position.value.toFloat().coerceIn(0f, duration.toFloat()),
            onValueChange = { onSeek(it.toDouble()) },
            valueRange = 0f..duration.toFloat().coerceAtLeast(1f),
            colors = SliderDefaults.colors(
                thumbColor = MusicDecodedColors.Primary,
                activeTrackColor = MusicDecodedColors.Primary,
                inactiveTrackColor = MusicDecodedColors.SurfaceSoft,
            ),
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            val style = MaterialTheme.typography.labelSmall.tabular()
            Text(formatDuration(position.value), style = style, color = MusicDecodedColors.TextSecondary)
            Text(formatDuration(duration), style = style, color = MusicDecodedColors.TextMuted)
        }
    }
}

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
