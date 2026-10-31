package com.musicdecoded.mobile.player

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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.core.isNoChord
import com.musicdecoded.mobile.core.upNext
import com.musicdecoded.mobile.ui.components.ChipRow
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.formatDuration
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.ui.components.Eyebrow
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles

/** A chord in the progression strip: its index in `analysis.chords` and length in beats. */
private data class StripChord(val index: Int, val beats: Int)

/**
 * Piano focus (DESIGN.md §3.7.1): the Piano learning view (§3.8) inline in the player's middle
 * region: current / up-next card, the large fixed keyboard (same voicing, inversion and fingering
 * rules; swipe for inversions), progression strip and learning controls. Scrolls if it must.
 */
@Composable
internal fun PianoFocusView(state: PlayerUiState, analysis: Analysis, position: State<Double>, viewModel: PlayerViewModel, modifier: Modifier = Modifier) {
    val chords = state.chords
    val chordIndex by remember(analysis) { derivedStateOf { currentState(position.value, analysis).chordIndex } }
    val next by remember(analysis) { derivedStateOf { upNext(position.value, analysis) } }
    val currentLabel = chordIndex?.let { analysis.chords[it].label }
    // Remembered so the keyboard skips the per-beat recomposition (up next counts beats) unless the voicing changes.
    val voicing = remember(chords, currentLabel, state.inversion) { currentLabel?.let { chords.voicing(it, state.inversion) } }
    val description = remember(chords, currentLabel, state.inversion) { currentLabel?.let { chords.noteNames(it, state.inversion) } }

    Column(
        modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        HeadCard(
            current = currentLabel,
            next = next?.let { analysis.chords[it.chordIndex].label },
            inBeats = next?.inBeats,
            chords = chords,
            inversion = state.inversion,
            modifier = Modifier.padding(horizontal = Spacing.gutter).padding(top = Spacing.xs),
        )
        FixedPianoKeyboard(
            voicing = voicing,
            noteName = chords::noteName,
            keysHeight = PianoKeyboardDefaults.LargeHeight,
            onStepInversion = viewModel::stepInversion,
            inversion = voicing?.let { state.inversion.coerceAtMost(it.inversions - 1) } ?: 0,
            showNoteNames = state.showNoteNames,
            description = description,
            // Inset by the gutter so the frame's lime shadowLg stays on screen (DESIGN.md §1).
            modifier = Modifier.padding(horizontal = Spacing.gutter),
            frameShadow = Shadows.lg,
        )
        ProgressionStrip(analysis, chordIndex, chords, onSeek = viewModel::seekTo)
        LearningControls(state, viewModel, voicing?.inversions ?: 0, Modifier.padding(horizontal = Spacing.gutter).padding(bottom = Spacing.md))
    }
}

/** CURRENT CHORD (name + notes) | UP NEXT (name + "in N beats"). */
@Composable
private fun HeadCard(current: String?, next: String?, inBeats: Int?, chords: ChordDisplay, inversion: Int, modifier: Modifier = Modifier) {
    Row(
        modifier
            .fillMaxWidth()
            .height(IntrinsicSize.Min)
            .block(MusicDecodedColors.Surface, shadow = Shadows.lg, shadowColor = MusicDecodedColors.Pink)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Eyebrow("Current chord")
            Text(
                current?.let(chords::name)?.ifEmpty { null } ?: "—",
                style = MaterialTheme.typography.displaySmall,
                color = MusicDecodedColors.Chord,
            )
            Text(current?.let { chords.noteNames(it, inversion) }.orEmpty(), style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        Box(Modifier.fillMaxHeight().width(Borders.separator).background(MusicDecodedColors.Line))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Eyebrow("Up next", color = MusicDecodedColors.TextMuted)
            Text(next?.let(chords::name) ?: "—", style = MaterialTheme.typography.displaySmall, color = MusicDecodedColors.Text)
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
        contentPadding = PaddingValues(horizontal = Spacing.gutter, vertical = Spacing.xxs),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        itemsIndexed(strip, key = { _, item -> item.index }) { position, item ->
            val chord = analysis.chords[item.index]
            val isCurrent = position == current
            Column(
                Modifier
                    .block(
                        if (isCurrent) MusicDecodedColors.Chord else MusicDecodedColors.Surface,
                        shadow = if (isCurrent) Shadows.sm else 0.dp,
                        shadowColor = MusicDecodedColors.Violet,
                        border = if (isCurrent) MusicDecodedColors.Border else MusicDecodedColors.BorderOnInk,
                        onClick = { onSeek(chord.start) },
                        onClickLabel = "Play from here",
                    )
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

/** Speed · Loop · Note names · Transpose, then Voicing. */
@Composable
private fun LearningControls(state: PlayerUiState, viewModel: PlayerViewModel, inversions: Int, modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        ChipRow {
            SpeedChip(state.speed, viewModel::setSpeed, Modifier.weight(1f))
            ControlChip(
                "Loop",
                state.loop?.let { "${formatDuration(it.start)}–${formatDuration(it.endExclusive)}" } ?: "Off",
                onClick = viewModel::toggleLoop,
                Modifier.weight(1f),
            )
        }
        ChipRow {
            ControlChip("Note names", if (state.showNoteNames) "On" else "Off", onClick = viewModel::toggleNoteNames, Modifier.weight(1f))
            TransposeChip(state.transpose, viewModel::setTranspose, Modifier.weight(1f))
        }
        VoicingChip(state.inversion, inversions, viewModel::setInversion, Modifier.fillMaxWidth())
    }
}
