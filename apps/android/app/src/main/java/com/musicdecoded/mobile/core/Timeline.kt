package com.musicdecoded.mobile.core

// Player logic from DESIGN.md §4. All times are seconds; all lookups are binary searches
// over the sorted `chords` / `beats` lists.

/** "Previous chord" restarts the current chord when the playhead is more than this far into it. */
const val PREVIOUS_CHORD_RESTART_THRESHOLD: Double = 1.0

/** Where the playhead is in an analysis. Indices point into [Analysis.chords] / [Analysis.beats]. */
data class PlaybackState(
    /** The segment with `start ≤ t < end`; `null` before the first or at/after the last `end`. */
    val chordIndex: Int?,
    /** The last beat with `t ≤ playhead`; `null` before the first beat. */
    val beatIndex: Int?,
    /** 0-based bar number, counted in downbeats; `null` before the first downbeat (pickup). */
    val bar: Int?,
)

/** The next real chord (never `N`/`X`) and how many beats until it starts. */
data class UpNext(
    /** Index into [Analysis.chords]. */
    val chordIndex: Int,
    /** Number of beats with `playhead < beat.t ≤ chord.start`. */
    val inBeats: Int,
)

/** Current chord, beat and bar at time [t]. */
fun currentState(t: Double, analysis: Analysis): PlaybackState {
    val chords = analysis.chords
    val index = upperBound(chords, t) { it.start } - 1
    val chordIndex = if (index < 0 || t >= chords[index].end) null else index

    val beatCount = upperBound(analysis.beats, t) { it.t }
    var downbeats = 0
    for (i in 0 until beatCount) if (analysis.beats[i].downbeat) downbeats++
    return PlaybackState(
        chordIndex = chordIndex,
        beatIndex = if (beatCount > 0) beatCount - 1 else null,
        bar = if (downbeats > 0) downbeats - 1 else null,
    )
}

/** The next segment after the one at [t] whose label isn't `N`/`X`, and the beats until it. */
fun upNext(t: Double, analysis: Analysis): UpNext? {
    val chords = analysis.chords
    val index = (upperBound(chords, t) { it.start } until chords.size)
        .firstOrNull { !isNoChord(chords[it].label) } ?: return null
    val beats = analysis.beats
    val inBeats = upperBound(beats, chords[index].start) { it.t } - upperBound(beats, t) { it.t }
    return UpNext(index, inBeats)
}

/**
 * Seek target for "previous chord": the current chord's start if more than 1 s into it,
 * otherwise the start of the previous non-`N` segment. `null` when there is none.
 */
fun previousChordStart(t: Double, analysis: Analysis): Double? {
    val chords = analysis.chords
    val current = upperBound(chords, t) { it.start } - 1
    if (current < 0) return null
    val chord = chords[current]
    if (!isNoChord(chord.label) && t - chord.start > PREVIOUS_CHORD_RESTART_THRESHOLD) {
        return chord.start
    }
    return chords.subList(0, current).lastOrNull { !isNoChord(it.label) }?.start
}

/** Seek target for "next chord": the start of the next non-`N` segment, or `null`. */
fun nextChordStart(t: Double, analysis: Analysis): Double? {
    val chords = analysis.chords
    val after = upperBound(chords, t) { it.start }
    return chords.subList(after, chords.size).firstOrNull { !isNoChord(it.label) }?.start
}

/**
 * The bar containing [t] (downbeat to next downbeat), for "loop current bar".
 * Before the first downbeat it is the pickup `0..<firstDownbeat`; after the last one it ends
 * at `meta.durationSec`. `null` if the analysis has no downbeats or [t] is outside `0..<durationSec`.
 */
fun barRange(t: Double, analysis: Analysis): OpenEndRange<Double>? {
    if (t < 0 || t >= analysis.meta.durationSec) return null
    val downbeats = analysis.beats.filter { it.downbeat }.map { it.t }
    if (downbeats.isEmpty()) return null
    val after = upperBound(downbeats, t) { it }
    val start = if (after == 0) 0.0 else downbeats[after - 1]
    val end = if (after < downbeats.size) downbeats[after] else analysis.meta.durationSec
    return if (start < end) start..<end else null
}

/** Index of the first element whose [key] is greater than [t] (elements must be sorted by [key]). */
internal inline fun <T> upperBound(elements: List<T>, t: Double, key: (T) -> Double): Int {
    var low = 0
    var high = elements.size
    while (low < high) {
        val mid = (low + high) ushr 1
        if (key(elements[mid]) <= t) low = mid + 1 else high = mid
    }
    return low
}
