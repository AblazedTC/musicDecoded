package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.upperBound
import kotlin.math.abs

// Pure timeline geometry (DESIGN.md §3.7 item 3): the playhead is fixed at the horizontal center
// and content scrolls under it, so x = center + (t − playhead) × pxPerSec.

/** Zoom limits in dp per second (DESIGN.md §3.7: 60–400, default 120). */
const val MIN_ZOOM_DP_PER_SEC = 60f
const val MAX_ZOOM_DP_PER_SEC = 400f
const val DEFAULT_ZOOM_DP_PER_SEC = 120f

fun clampZoom(dpPerSec: Float): Float = dpPerSec.coerceIn(MIN_ZOOM_DP_PER_SEC, MAX_ZOOM_DP_PER_SEC)

/** Maps time ↔ x for a viewport of [width] px whose center is the playhead at [playhead] seconds. */
data class TimelineViewport(val width: Float, val playhead: Double, val pxPerSec: Float) {
    val center: Float get() = width / 2f
    val startTime: Double get() = playhead - center / pxPerSec
    val endTime: Double get() = playhead + center / pxPerSec

    fun x(t: Double): Float = (center + (t - playhead) * pxPerSec).toFloat()
    fun time(x: Float): Double = playhead + (x - center) / pxPerSec
}

/** Indices of the chords that overlap `[from, to]`, so only the visible window is drawn. */
fun visibleChordIndices(chords: List<Analysis.Chord>, from: Double, to: Double): IntRange {
    if (chords.isEmpty()) return IntRange.EMPTY
    val first = (upperBound(chords, from) { it.start } - 1).coerceAtLeast(0)
    val last = upperBound(chords, to) { it.start } - 1
    return first..last
}

/** Indices of the beats with `from ≤ t ≤ to`. */
fun visibleBeatIndices(beats: List<Analysis.Beat>, from: Double, to: Double): IntRange {
    val first = upperBound(beats, from - 1e-9) { it.t }
    val last = upperBound(beats, to) { it.t } - 1
    return first..last
}

/** Indices of the waveform peaks (one per `1/fps` s) inside `[from, to]`. */
fun visiblePeakIndices(peaks: Analysis.WaveformPeaks, from: Double, to: Double): IntRange {
    if (peaks.peaks.isEmpty() || peaks.fps <= 0 || to < 0 || from * peaks.fps >= peaks.peaks.size) return IntRange.EMPTY
    val first = (from * peaks.fps).toInt().coerceIn(0, peaks.peaks.size - 1)
    val last = (to * peaks.fps).toInt().coerceIn(0, peaks.peaks.size - 1)
    return first..last
}

/** The beat time nearest to [t]; [t] itself when there are no beats. */
fun snapToBeat(t: Double, beats: List<Analysis.Beat>): Double {
    if (beats.isEmpty()) return t
    val after = upperBound(beats, t) { it.t }
    val candidates = listOfNotNull(beats.getOrNull(after - 1)?.t, beats.getOrNull(after)?.t)
    return candidates.minBy { abs(it - t) }
}

/**
 * A custom loop dragged between [a] and [b] (either order), snapped to beats (DESIGN.md §4).
 * A drag that snaps to a single beat extends to the next beat. `null` if nothing usable is left.
 */
fun snappedLoop(a: Double, b: Double, analysis: Analysis): OpenEndRange<Double>? {
    val duration = analysis.meta.durationSec
    var start = snapToBeat(minOf(a, b), analysis.beats).coerceIn(0.0, duration)
    var end = snapToBeat(maxOf(a, b), analysis.beats).coerceIn(0.0, duration)
    if (end <= start) {
        end = analysis.beats.firstOrNull { it.t > start }?.t ?: duration
    }
    if (end <= start) start = (end - 1.0).coerceAtLeast(0.0)
    return if (start < end) start..<end else null
}

/** Seconds between time labels so they are at least [minSpacingPx] apart. */
fun timeLabelStep(pxPerSec: Float, minSpacingPx: Float): Int =
    LABEL_STEPS.firstOrNull { it * pxPerSec >= minSpacingPx } ?: LABEL_STEPS.last()

private val LABEL_STEPS = listOf(1, 2, 5, 10, 15, 30, 60)
