package com.musicdecoded.mobile.player

import android.content.res.Configuration
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.SkipNext
import androidx.compose.material.icons.rounded.SkipPrevious
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.core.isNoChord
import com.musicdecoded.mobile.core.upNext
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles

/** Octaves on the large keyboard: 2 in portrait, wider in landscape (DESIGN.md §3.8). */
private const val PORTRAIT_OCTAVES = 2
private const val LANDSCAPE_OCTAVES = 4

/** A chord in the progression strip: its index in `analysis.chords` and length in beats. */
private data class StripChord(val index: Int, val beats: Int)

/** Piano learning view (DESIGN.md §3.8). Shares the player's [PlayerViewModel]. */
@Composable
fun PianoScreen(viewModel: PlayerViewModel, onBack: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val position = viewModel.position.collectAsStateWithLifecycle()
    PlaybackClock(viewModel, state.isPlaying)
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE

    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        val track = state.track
        PlayerHeader(track?.title.orEmpty(), track?.artist, onBack = onBack)
        if (track == null) return@Column
        val analysis = track.analysis
        val chords = state.chords
        val chordIndex by remember(analysis) { derivedStateOf { currentState(position.value, analysis).chordIndex } }
        val next by remember(analysis) { derivedStateOf { upNext(position.value, analysis) } }
        val currentLabel = chordIndex?.let { analysis.chords[it].label }

        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            HeadCard(
                current = currentLabel,
                next = next?.let { analysis.chords[it.chordIndex].label },
                inBeats = next?.inBeats,
                chords = chords,
                modifier = Modifier.padding(horizontal = Spacing.gutter),
            )
            Box(Modifier.padding(horizontal = Spacing.gutter)) {
                SeekBar(position, analysis.meta.durationSec, viewModel::seekTo)
            }
            PianoKeyboard(
                notes = currentLabel?.let(chords::notes).orEmpty(),
                bassNote = currentLabel?.let(chords::bassNote),
                octaves = if (landscape) LANDSCAPE_OCTAVES else PORTRAIT_OCTAVES,
                height = PianoKeyboardDefaults.LargeHeight,
                noteName = chords::noteName.takeIf { state.showNoteNames },
            )
            ProgressionStrip(analysis, chordIndex, chords, onSeek = viewModel::seekTo)
            LearningControls(state, viewModel, Modifier.padding(horizontal = Spacing.gutter))
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(Spacing.xl, Alignment.CenterHorizontally),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                MdIconButton(Icons.Rounded.SkipPrevious, contentDescription = "Previous chord", onClick = viewModel::previousChord)
                PlayButton(state.isPlaying, onClick = viewModel::togglePlay)
                MdIconButton(Icons.Rounded.SkipNext, contentDescription = "Next chord", onClick = viewModel::nextChord)
            }
            if (!landscape) {
                Text(
                    "Rotate your device for a wider keyboard",
                    style = MaterialTheme.typography.labelSmall,
                    color = MusicDecodedColors.TextMuted,
                    modifier = Modifier.align(Alignment.CenterHorizontally).padding(bottom = Spacing.md),
                )
            }
        }
    }
}

/** CURRENT CHORD (name + notes) | UP NEXT (name + "in N beats"). */
@Composable
private fun HeadCard(current: String?, next: String?, inBeats: Int?, chords: ChordDisplay, modifier: Modifier = Modifier) {
    Row(
        modifier
            .fillMaxWidth()
            .height(IntrinsicSize.Min)
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text("CURRENT CHORD", style = TextStyles.overline)
            Text(current?.let(chords::name)?.ifEmpty { null } ?: "—", style = TextStyles.currentChord)
            Text(current?.let(chords::noteNames).orEmpty(), style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        Box(Modifier.fillMaxHeight().width(Sizes.hairline).background(MusicDecodedColors.Line))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text("UP NEXT", style = TextStyles.overline)
            Text(next?.let(chords::name) ?: "—", style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Text)
            if (inBeats != null) {
                Text(
                    if (inBeats == 1) "in 1 beat" else "in $inBeats beats",
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.TextSecondary,
                )
            }
        }
    }
}

/** Horizontally scrolling chord cards (chord + length in beats); the current one is filled and kept in view. */
@Composable
private fun ProgressionStrip(analysis: Analysis, chordIndex: Int?, chords: ChordDisplay, onSeek: (Double) -> Unit) {
    val strip = remember(analysis) {
        analysis.chords.withIndex()
            .filter { !isNoChord(it.value.label) }
            .map { (index, chord) -> StripChord(index, analysis.beats.count { it.t >= chord.start && it.t < chord.end }) }
    }
    val current = strip.indexOfFirst { it.index == chordIndex }
    val listState = rememberLazyListState()
    LaunchedEffect(current) {
        if (current >= 0) listState.animateScrollToItem((current - 1).coerceAtLeast(0))
    }
    LazyRow(
        state = listState,
        contentPadding = PaddingValues(horizontal = Spacing.gutter),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        itemsIndexed(strip, key = { _, item -> item.index }) { position, item ->
            val chord = analysis.chords[item.index]
            val isCurrent = position == current
            Column(
                Modifier
                    .clip(Shapes.chip)
                    .background(if (isCurrent) MusicDecodedColors.Chord else MusicDecodedColors.Surface)
                    .border(Sizes.hairline, if (isCurrent) MusicDecodedColors.Chord else MusicDecodedColors.Line, Shapes.chip)
                    .clickable(role = Role.Button, onClickLabel = "Play from here") { onSeek(chord.start) }
                    .padding(horizontal = Spacing.sm, vertical = Spacing.xs),
            ) {
                Text(
                    chords.name(chord.label),
                    style = MaterialTheme.typography.titleSmall,
                    color = if (isCurrent) MusicDecodedColors.OnChord else MusicDecodedColors.Chord,
                )
                Text(
                    if (item.beats == 1) "1 beat" else "${item.beats} beats",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (isCurrent) MusicDecodedColors.OnChord else MusicDecodedColors.TextMuted,
                )
            }
        }
    }
}

/** Speed · Loop · Note names · Transpose (2×2). */
@Composable
private fun LearningControls(state: PlayerUiState, viewModel: PlayerViewModel, modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            SpeedChip(state.speed, viewModel::setSpeed, Modifier.weight(1f))
            ControlChip(
                "Loop",
                state.loop?.let { "${formatDuration(it.start)}–${formatDuration(it.endExclusive)}" } ?: "Off",
                onClick = viewModel::toggleLoop,
                Modifier.weight(1f),
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            ControlChip("Note names", if (state.showNoteNames) "On" else "Off", onClick = viewModel::toggleNoteNames, Modifier.weight(1f))
            TransposeChip(state.transpose, viewModel::setTranspose, Modifier.weight(1f))
        }
    }
}

@Preview(heightDp = 900)
@Composable
private fun PianoScreenPreview() = MusicDecodedTheme {
    val analysis = rememberPreviewAnalysis() ?: return@MusicDecodedTheme
    val context = LocalContext.current
    val viewModel = remember {
        PlayerViewModel(
            loadTrack = { PlayerTrack(analysis, "Sunny Days", "Fixture", favorite = false, audio = null) },
            voicings = runCatching { loadChordVoicings(context) }.getOrNull(),
            createEngine = { SilentPlaybackEngine(it.analysis.meta.durationSec) },
        )
    }
    PianoScreen(viewModel, onBack = {})
}
