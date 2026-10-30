package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.isNoChord
import com.musicdecoded.mobile.core.upperBound
import kotlin.math.abs

/**
 * One beat cell in the Beat grid (DESIGN.md §3.7.1). [beat] = index in `analysis.beats`, or −1 for
 * a padding cell (pickup / last bar). [chord] = the chord sounding on this beat (`null` = none);
 * [starts] = it changes on this beat (the cell shows its name; otherwise a continuation underline).
 */
data class GridCell(val beat: Int, val chord: Int?, val starts: Boolean)

/** A bar of [GridCell]s (always `timeSignature.numerator` cells), numbered from 1, covering [start, end). */
data class GridBar(val number: Int, val cells: List<GridCell>, val start: Double, val end: Double) {
    val beats: IntRange get() = cells.first { it.beat >= 0 }.beat..cells.last { it.beat >= 0 }.beat
}

/** A chord box in the Chord sheet: [chord] index, where tapping seeks, and the beats it covers in its bar. */
data class SheetChord(val chord: Int, val seekTo: Double, val beats: IntRange)

data class SheetBar(val number: Int, val chords: List<SheetChord>, val start: Double, val end: Double)

/** Chord sheet: 4 bars per line. */
const val SHEET_BARS_PER_LINE = 4

/** Beat whose time is nearest [t] (`-1` when there are no beats). */
fun nearestBeat(t: Double, beats: List<Analysis.Beat>): Int {
    if (beats.isEmpty()) return -1
    val after = upperBound(beats, t) { it.t }.coerceAtMost(beats.lastIndex)
    val before = (after - 1).coerceAtLeast(0)
    return if (abs(beats[before].t - t) <= abs(beats[after].t - t)) before else after
}

/** The beat under the playhead (`-1` before the first beat). */
fun beatAt(t: Double, beats: List<Analysis.Beat>): Int = upperBound(beats, t) { it.t } - 1

/**
 * The beat grid: bars split at downbeats (or every `numerator` beats when there are none, or when a
 * bar runs long), the first and last bars padded to `numerator` cells. Each chord change is mapped to
 * its nearest beat; if two land on one beat the last real chord wins. Empty when there are no beats.
 */
fun beatGrid(analysis: Analysis): List<GridBar> {
    val beats = analysis.beats
    if (beats.isEmpty()) return emptyList()
    val perBar = analysis.timeSignature?.numerator?.takeIf { it > 0 } ?: DEFAULT_BEATS_PER_BAR

    val startsAt = HashMap<Int, Int>()
    analysis.chords.forEachIndexed { index, chord ->
        val beat = nearestBeat(chord.start, beats)
        val existing = startsAt[beat]
        if (existing == null || !isNoChord(chord.label) || isNoChord(analysis.chords[existing].label)) startsAt[beat] = index
    }
    var holding: Int? = null
    val cells = beats.indices.map { i ->
        val start = startsAt[i]
        if (start != null) holding = start.takeUnless { isNoChord(analysis.chords[it].label) }
        GridCell(i, holding, starts = start != null && holding != null)
    }

    val hasDownbeats = beats.any { it.downbeat }
    val groups = ArrayList<MutableList<GridCell>>()
    for (cell in cells) {
        val current = groups.lastOrNull()
        val newBar = current == null || current.size >= perBar || (hasDownbeats && beats[cell.beat].downbeat)
        if (newBar) groups += mutableListOf(cell) else current!!.add(cell)
    }
    val padding = GridCell(-1, null, false)
    return groups.mapIndexed { index, group ->
        val padded = when {
            group.size >= perBar -> group
            index == 0 -> List(perBar - group.size) { padding } + group
            else -> group + List(perBar - group.size) { padding }
        }
        val end = groups.getOrNull(index + 1)?.first()?.let { beats[it.beat].t } ?: analysis.meta.durationSec
        GridBar(index + 1, padded, beats[group.first().beat].t, end)
    }
}

/**
 * The chord sheet: per bar, the chords sounding on its beats in order, a chord repeated within the
 * bar shown once. Without beats each chord is its own "bar". `N`/`X` never get a box.
 */
fun chordSheet(analysis: Analysis, grid: List<GridBar> = beatGrid(analysis)): List<SheetBar> {
    if (grid.isEmpty()) {
        return analysis.chords.withIndex().filter { !isNoChord(it.value.label) }.mapIndexed { n, (index, chord) ->
            SheetBar(n + 1, listOf(SheetChord(index, chord.start, IntRange.EMPTY)), chord.start, chord.end)
        }
    }
    val beats = analysis.beats
    return grid.map { bar ->
        val chords = ArrayList<SheetChord>()
        for (cell in bar.cells) {
            val chord = cell.chord ?: continue
            val last = chords.lastOrNull()
            if (last != null && analysis.chords[last.chord].label == analysis.chords[chord].label) {
                chords[chords.lastIndex] = last.copy(beats = last.beats.first..cell.beat)
            } else {
                // The bar's first beat seeks to the bar start; a change seeks to the chord itself.
                val seek = if (cell.starts) analysis.chords[chord].start else beats[cell.beat].t
                chords += SheetChord(chord, seek, cell.beat..cell.beat)
            }
        }
        SheetBar(bar.number, chords, bar.start, bar.end)
    }
}

/** Index of the bar holding [beat] (`-1` if none). */
fun barOfBeat(grid: List<GridBar>, beat: Int): Int = grid.indexOfFirst { beat in it.beats }

private const val DEFAULT_BEATS_PER_BAR = 4
