package com.musicdecoded.mobile.player

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.text.BasicText
import androidx.compose.foundation.text.TextAutoSize
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.ui.components.hardShadow
import com.musicdecoded.mobile.ui.theme.ArchivoBlack
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.SpaceGrotesk
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.tabular

/** Sizes for the Beat grid and Chord sheet (component sizes, not spacing). */
private object GridMetrics {
    val cellHeight = 48.dp
    val barNumber = 28.dp
    val underline = 2.dp
    val name = 16.sp
    val minName = 11.sp
    val numeral = 12.sp
}

private val nameStyle = TextStyle(fontFamily = ArchivoBlack, fontSize = GridMetrics.name)
private val numeralStyle = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = GridMetrics.numeral)

/**
 * Beat grid (DESIGN.md §3.7.1): one row per bar, one cell per beat, bar numbers on the left. A
 * chord's name (+ numeral) sits on the beat it starts; later beats get a thin lime underline.
 * Current beat = lime fill, current bar = lime outline, kept in the upper third. Tap = seek.
 */
@Composable
fun BeatGridView(analysis: Analysis, chords: ChordDisplay, position: State<Double>, onSeek: (Double) -> Unit, modifier: Modifier = Modifier) {
    val grid = remember(analysis) { beatGrid(analysis) }
    if (grid.isEmpty()) {
        Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("No beat grid for this song", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        return
    }
    val currentBeat by remember(analysis) { derivedStateOf { beatAt(position.value, analysis.beats) } }
    val currentBar by remember(grid) { derivedStateOf { barOfBeat(grid, currentBeat) } }
    val listState = rememberLazyListState()
    AutoScroll(listState, currentBar)

    LazyColumn(modifier.fillMaxSize(), state = listState, contentPadding = PaddingValues(Spacing.gutter, Spacing.xxs), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
        itemsIndexed(grid, key = { _, bar -> bar.number }) { index, bar ->
            // The current beat if it's in this bar: only the bars whose highlight changes recompose on a beat.
            val beatHere by remember(index) { derivedStateOf { currentBeat.takeIf { currentBar == index } } }
            val isCurrentBar = beatHere != null
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    bar.number.toString(),
                    style = MaterialTheme.typography.labelSmall.tabular(),
                    color = if (isCurrentBar) MusicDecodedColors.Primary else MusicDecodedColors.TextMuted,
                    modifier = Modifier.width(GridMetrics.barNumber),
                )
                Row(
                    Modifier
                        .weight(1f)
                        .border(Borders.thick, if (isCurrentBar) MusicDecodedColors.Primary else MusicDecodedColors.BorderOnInk)
                        .padding(Borders.thick),
                ) {
                    bar.cells.forEach { cell -> BeatCell(cell, analysis, chords, current = cell.beat >= 0 && cell.beat == beatHere, onSeek, Modifier.weight(1f)) }
                }
            }
        }
    }
}

@Composable
private fun BeatCell(cell: GridCell, analysis: Analysis, chords: ChordDisplay, current: Boolean, onSeek: (Double) -> Unit, modifier: Modifier) {
    val label = cell.chord?.let { analysis.chords[it].label }
    val ink = if (current) MusicDecodedColors.OnChord else MusicDecodedColors.Text
    Box(
        modifier
            .height(GridMetrics.cellHeight)
            .background(if (current) MusicDecodedColors.Chord else MusicDecodedColors.Surface)
            .border(Borders.separator / 2, MusicDecodedColors.Ink)
            .then(
                if (cell.beat >= 0) {
                    Modifier.clickable(role = Role.Button, onClickLabel = "Play from this beat") { onSeek(analysis.beats[cell.beat].t) }
                        .semantics { contentDescription = listOfNotNull("Beat ${cell.beat + 1}", label?.let(chords::name)?.takeIf { cell.starts }).joinToString(", ") }
                } else Modifier,
            ),
        contentAlignment = Alignment.Center,
    ) {
        if (label != null && cell.starts) {
            ChordText(chords.name(label), chords.numeral(label), ink)
        } else if (label != null) {
            Box(
                Modifier
                    .align(Alignment.BottomCenter)
                    .padding(horizontal = Spacing.xxs, vertical = Spacing.xxs)
                    .fillMaxWidth()
                    .height(GridMetrics.underline)
                    .background(if (current) MusicDecodedColors.OnChord else MusicDecodedColors.Primary),
            )
        }
    }
}

/**
 * Chord sheet (DESIGN.md §3.7.1): the progression as boxes, 4 bars per line with bar lines between
 * them, a chord repeated within a bar shown once. Current chord = lime box; auto-scrolls; tap = seek.
 */
@Composable
fun ChordSheetView(analysis: Analysis, chords: ChordDisplay, position: State<Double>, onSeek: (Double) -> Unit, modifier: Modifier = Modifier) {
    val lines = remember(analysis) { chordSheet(analysis).chunked(SHEET_BARS_PER_LINE) }
    val hasBeats = analysis.beats.isNotEmpty()
    val currentBeat by remember(analysis) { derivedStateOf { beatAt(position.value, analysis.beats) } }
    val currentChord by remember(analysis) { derivedStateOf { currentState(position.value, analysis).chordIndex } }
    fun isCurrent(entry: SheetChord) = if (hasBeats) currentBeat in entry.beats else entry.chord == currentChord
    val currentLine by remember(lines) {
        derivedStateOf { lines.indexOfFirst { line -> line.any { bar -> bar.chords.any(::isCurrent) } } }
    }
    val listState = rememberLazyListState()
    AutoScroll(listState, currentLine)

    LazyColumn(modifier.fillMaxSize(), state = listState, contentPadding = PaddingValues(Spacing.gutter, Spacing.xs), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        itemsIndexed(lines) { _, line ->
            // Only the line(s) gaining or losing the current chord recompose on a beat.
            val current by remember(line) { derivedStateOf { line.flatMap { it.chords }.filter(::isCurrent) } }
            Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min)) {
                BarLine()
                for (slot in 0 until SHEET_BARS_PER_LINE) {
                    val bar = line.getOrNull(slot)
                    Row(
                        Modifier.weight(1f).padding(horizontal = Spacing.xxs),
                        horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        bar?.chords?.forEach { entry ->
                            SheetBox(analysis.chords[entry.chord].label, chords, entry in current, onClick = { onSeek(entry.seekTo) }, Modifier.weight(1f))
                        }
                        if (bar != null && bar.chords.isEmpty()) {
                            Text("—", style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextMuted, modifier = Modifier.heightIn(min = GridMetrics.cellHeight))
                        }
                    }
                    if (bar != null) BarLine()
                }
            }
        }
    }
}

@Composable
private fun BarLine() = Box(Modifier.fillMaxHeight().width(Borders.separator).background(MusicDecodedColors.TextMuted))

/** Same box as the timeline labels: square, 2dp black border; current = lime + ink + small hard shadow. */
@Composable
private fun SheetBox(label: String, chords: ChordDisplay, current: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier
            .heightIn(min = GridMetrics.cellHeight)
            .hardShadow(if (current) Shadows.sm else 0.dp, MusicDecodedColors.Violet)
            .background(if (current) MusicDecodedColors.Chord else MusicDecodedColors.SurfaceRaised)
            .border(Borders.separator, MusicDecodedColors.Border)
            .clickable(role = Role.Button, onClickLabel = "Play from here", onClick = onClick)
            // No side padding: two chords can share a quarter of the width.
            .padding(vertical = Spacing.xxs),
        contentAlignment = Alignment.Center,
    ) {
        ChordText(chords.name(label), chords.numeral(label), if (current) MusicDecodedColors.OnChord else MusicDecodedColors.Text)
    }
}

@Composable
private fun ChordText(name: String, numeral: String?, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        // Shrinks (down to the 11sp minimum) when several chords share a bar.
        BasicText(
            name,
            style = nameStyle.copy(color = color),
            maxLines = 1,
            softWrap = false,
            overflow = TextOverflow.Clip,
            autoSize = TextAutoSize.StepBased(minFontSize = GridMetrics.minName, maxFontSize = GridMetrics.name),
        )
        if (numeral != null) Text(numeral, style = numeralStyle, color = color, maxLines = 1, softWrap = false)
    }
}

/** Keeps item [index] in the upper third of the list as it changes. */
@Composable
private fun AutoScroll(listState: LazyListState, index: Int) {
    LaunchedEffect(index) {
        if (index < 0) return@LaunchedEffect
        val visible = listState.layoutInfo.visibleItemsInfo.size.coerceAtLeast(1)
        listState.animateScrollToItem((index - visible / 3).coerceAtLeast(0))
    }
}
