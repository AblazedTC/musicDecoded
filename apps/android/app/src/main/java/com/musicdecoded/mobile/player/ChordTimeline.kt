package com.musicdecoded.mobile.player

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.CustomAccessibilityAction
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.customActions
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.chordDisplayName
import com.musicdecoded.mobile.core.isNoChord
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Fixed drawing sizes of the timeline (component sizes, not spacing). */
private object TimelineMetrics {
    val labels = 16.dp
    val chordTrack = 32.dp
    val waveform = 72.dp
    val labelSpacing = 56.dp
    val barMaxWidth = 3.dp
    val playhead = 1.5.dp
    val playheadGlow = 6.dp
    val playheadDot = 4.dp
    val blockRadius = 6.dp
    val minLabelWidth = 16.dp
}

private enum class Gesture { NONE, SCRUB, LOOP, ZOOM, IGNORE }

/**
 * The player timeline (DESIGN.md §3.7 item 3): time labels, a chord track over the waveform, downbeat
 * gridlines and a fixed center playhead. Content scrolls under the playhead. Drag the waveform to
 * scrub, drag the chord track to select a loop (snapped to beats), tap a chord block to seek to its
 * start, pinch to zoom (60–400 dp/s). Only the visible window is drawn.
 *
 * [position] is read in the draw phase only, so a 60 Hz clock redraws without recomposing.
 */
@Composable
fun ChordTimeline(
    analysis: Analysis,
    position: () -> Double,
    chordName: (String) -> String,
    onSeek: (Double) -> Unit,
    modifier: Modifier = Modifier,
    loop: OpenEndRange<Double>? = null,
    onLoopSelected: (OpenEndRange<Double>) -> Unit = {},
    /** The current chord's display name, for screen readers. */
    currentChord: String? = null,
) {
    var zoom by rememberSaveable { mutableFloatStateOf(DEFAULT_ZOOM_DP_PER_SEC) }
    var loopDraft by remember { mutableStateOf<OpenEndRange<Double>?>(null) }
    val currentPosition by rememberUpdatedState(position)
    val currentAnalysis by rememberUpdatedState(analysis)
    val seek by rememberUpdatedState(onSeek)
    val selectLoop by rememberUpdatedState(onLoopSelected)

    val textMeasurer = rememberTextMeasurer(cacheSize = 64)
    val labelStyle = MaterialTheme.typography.labelSmall.copy(color = MusicDecodedColors.TextMuted, fontFeatureSettings = "tnum")
    val chordStyle = MaterialTheme.typography.labelLarge
    val activeChordStyle = remember(chordStyle) { chordStyle.copy(color = MusicDecodedColors.OnChord) }
    val idleChordStyle = remember(chordStyle) { chordStyle.copy(color = MusicDecodedColors.Chord) }

    Column(
        modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.TimelineBg, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.TimelineBorder, Shapes.card)
            .padding(vertical = Spacing.sm),
        verticalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Canvas(
            Modifier
                .fillMaxWidth()
                .height(with(LocalDensity.current) { (TimelineLayout.waveTop(this) + TimelineMetrics.waveform.toPx()).toDp() })
                .clipToBounds()
                // Canvas content is invisible to TalkBack: describe it, and offer zoom as actions.
                // Chords are reachable one by one through previous/next chord in the transport.
                .semantics {
                    contentDescription = "Chord timeline"
                    stateDescription = currentChord?.let { "Current chord $it" } ?: "No chord"
                    customActions = listOf(
                        CustomAccessibilityAction("Zoom in") { zoom = clampZoom(zoom * ZOOM_STEP); true },
                        CustomAccessibilityAction("Zoom out") { zoom = clampZoom(zoom / ZOOM_STEP); true },
                    )
                }
                .pointerInput(Unit) {
                    val chordTop = TimelineLayout.chordTop(this)
                    val chordBottom = chordTop + scaled(TimelineMetrics.chordTrack)
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        val pxPerSec = zoom * density
                        val startViewport = TimelineViewport(size.width.toFloat(), currentPosition(), pxPerSec)
                        val onChordTrack = down.position.y in chordTop..chordBottom
                        var gesture = Gesture.NONE
                        var lastSpan = 0f
                        do {
                            val event = awaitPointerEvent()
                            val pressed = event.changes.filter { it.pressed }
                            if (pressed.size >= 2 && gesture != Gesture.IGNORE) {
                                val span = hypot(pressed[0].position.x - pressed[1].position.x, pressed[0].position.y - pressed[1].position.y)
                                if (gesture == Gesture.ZOOM && lastSpan > 0f) zoom = clampZoom(zoom * span / lastSpan)
                                gesture = Gesture.ZOOM
                                loopDraft = null
                                lastSpan = span
                                event.changes.forEach { it.consume() }
                            } else if (pressed.size == 1 && gesture != Gesture.ZOOM && gesture != Gesture.IGNORE) {
                                val change = pressed[0]
                                val delta = change.position - down.position
                                if (gesture == Gesture.NONE && delta.getDistance() > viewConfiguration.touchSlop) {
                                    // Vertical drags belong to the screen's scroll.
                                    gesture = when {
                                        kotlin.math.abs(delta.y) > kotlin.math.abs(delta.x) -> Gesture.IGNORE
                                        onChordTrack -> Gesture.LOOP
                                        else -> Gesture.SCRUB
                                    }
                                }
                                val duration = currentAnalysis.meta.durationSec
                                when (gesture) {
                                    Gesture.SCRUB -> seek((startViewport.playhead - delta.x / pxPerSec).coerceIn(0.0, duration))
                                    Gesture.LOOP -> loopDraft = snappedLoop(
                                        startViewport.time(down.position.x),
                                        startViewport.time(change.position.x),
                                        currentAnalysis,
                                    )
                                    else -> Unit
                                }
                                if (gesture == Gesture.SCRUB || gesture == Gesture.LOOP) change.consume()
                            }
                        } while (event.changes.any { it.pressed })

                        when (gesture) {
                            Gesture.NONE -> if (onChordTrack) {
                                val t = startViewport.time(down.position.x)
                                currentAnalysis.chords
                                    .firstOrNull { t >= it.start && t < it.end && !isNoChord(it.label) }
                                    ?.let { seek(it.start) }
                            }
                            Gesture.LOOP -> loopDraft?.let(selectLoop)
                            else -> Unit
                        }
                        loopDraft = null
                    }
                },
        ) {
            val viewport = TimelineViewport(size.width, currentPosition(), zoom * density)
            val layout = TimelineLayout(this)
            drawTimeLabels(viewport, analysis, layout, textMeasurer, labelStyle)
            drawDownbeats(viewport, analysis, layout)
            drawWaveform(viewport, analysis, layout)
            drawChordBlocks(viewport, analysis, layout, chordName, textMeasurer, activeChordStyle, idleChordStyle)
            (loopDraft ?: loop)?.let { drawLoop(viewport, it, layout) }
            drawPlayhead(viewport, layout)
        }
        Text(
            "Pinch to zoom · drag to scrub",
            style = MaterialTheme.typography.labelSmall,
            color = MusicDecodedColors.TextMuted,
            modifier = Modifier.padding(horizontal = Spacing.md),
        )
    }
}

/** A band that holds text: [height] at 100% font size, growing with the user's font scale (T-503a). */
private fun Density.scaled(height: Dp): Float = max(height.toPx(), height.value.sp.toPx())

/** One accessibility zoom step (×1.5 / ÷1.5). */
private const val ZOOM_STEP = 1.5f

/** Vertical bands of the canvas, in px. */
private class TimelineLayout(scope: DrawScope) {
    val labelsBottom: Float
    val chordTop: Float
    val chordBottom: Float
    val waveTop: Float
    val waveBottom: Float

    init {
        with(scope) {
            labelsBottom = scaled(TimelineMetrics.labels)
            chordTop = labelsBottom + Spacing.xxs.toPx()
            chordBottom = chordTop + scaled(TimelineMetrics.chordTrack)
            waveTop = chordBottom + Spacing.xs.toPx()
            waveBottom = waveTop + TimelineMetrics.waveform.toPx()
        }
    }

    companion object {
        fun chordTop(density: Density): Float = with(density) { scaled(TimelineMetrics.labels) + Spacing.xxs.toPx() }

        fun waveTop(density: Density): Float =
            with(density) { chordTop(density) + scaled(TimelineMetrics.chordTrack) + Spacing.xs.toPx() }
    }
}

private fun DrawScope.drawTimeLabels(
    viewport: TimelineViewport,
    analysis: Analysis,
    layout: TimelineLayout,
    textMeasurer: TextMeasurer,
    style: TextStyle,
) {
    val step = timeLabelStep(viewport.pxPerSec, TimelineMetrics.labelSpacing.toPx())
    var second = (ceil(max(0.0, viewport.startTime) / step) * step).toInt()
    val last = min(viewport.endTime, analysis.meta.durationSec)
    while (second <= last) {
        val text = textMeasurer.measure(formatDuration(second.toDouble()), style)
        val x = viewport.x(second.toDouble())
        drawText(text, topLeft = Offset(x - text.size.width / 2f, (layout.labelsBottom - text.size.height) / 2f))
        drawLine(MusicDecodedColors.Line, Offset(x, layout.labelsBottom - Spacing.xxs.toPx() / 2), Offset(x, layout.labelsBottom), Sizes.hairline.toPx())
        second += step
    }
}

private fun DrawScope.drawDownbeats(viewport: TimelineViewport, analysis: Analysis, layout: TimelineLayout) {
    val color = MusicDecodedColors.Line.copy(alpha = 0.5f)
    val stroke = Sizes.hairline.toPx()
    for (i in visibleBeatIndices(analysis.beats, viewport.startTime, viewport.endTime)) {
        val beat = analysis.beats[i]
        if (!beat.downbeat) continue
        val x = viewport.x(beat.t)
        drawLine(color, Offset(x, layout.chordTop), Offset(x, layout.waveBottom), stroke)
    }
}

/** Waveform bars from `waveformPeaks` (played = primary), or beat ticks when there are no peaks. */
private fun DrawScope.drawWaveform(viewport: TimelineViewport, analysis: Analysis, layout: TimelineLayout) {
    val middle = (layout.waveTop + layout.waveBottom) / 2f
    val height = layout.waveBottom - layout.waveTop
    val minBar = Sizes.hairline.toPx()
    val peaks = analysis.waveformPeaks
    if (peaks != null && peaks.peaks.isNotEmpty() && peaks.fps > 0) {
        val pitch = (viewport.pxPerSec / peaks.fps).toFloat()
        val barWidth = (pitch * 0.6f).coerceIn(minBar, TimelineMetrics.barMaxWidth.toPx())
        for (i in visiblePeakIndices(peaks, viewport.startTime, viewport.endTime)) {
            val t = i / peaks.fps
            val x = viewport.x(t)
            val half = max(minBar, peaks.peaks[i].toFloat() * height) / 2f
            val color = if (t < viewport.playhead) MusicDecodedColors.WavePlayed else MusicDecodedColors.WaveIdle
            drawLine(color, Offset(x, middle - half), Offset(x, middle + half), barWidth, StrokeCap.Round)
        }
    } else {
        val stroke = Sizes.hairline.toPx() * 2
        for (i in visibleBeatIndices(analysis.beats, viewport.startTime, viewport.endTime)) {
            val beat = analysis.beats[i]
            val x = viewport.x(beat.t)
            val half = height * (if (beat.downbeat) 0.35f else 0.18f)
            val color = if (beat.t < viewport.playhead) MusicDecodedColors.WavePlayed else MusicDecodedColors.WaveIdle
            drawLine(color, Offset(x, middle - half), Offset(x, middle + half), stroke, StrokeCap.Round)
        }
    }
}

private fun DrawScope.drawChordBlocks(
    viewport: TimelineViewport,
    analysis: Analysis,
    layout: TimelineLayout,
    chordName: (String) -> String,
    textMeasurer: TextMeasurer,
    activeStyle: TextStyle,
    idleStyle: TextStyle,
) {
    val inset = Spacing.xxs.toPx() / 2
    val radius = CornerRadius(TimelineMetrics.blockRadius.toPx())
    val padding = Spacing.xs.toPx()
    val border = Sizes.hairline.toPx()
    val height = layout.chordBottom - layout.chordTop
    for (i in visibleChordIndices(analysis.chords, viewport.startTime, viewport.endTime)) {
        val chord = analysis.chords[i]
        if (isNoChord(chord.label)) continue
        val left = viewport.x(chord.start) + inset
        val width = viewport.x(chord.end) - inset - left
        if (width <= 0f) continue
        val current = viewport.playhead >= chord.start && viewport.playhead < chord.end
        val topLeft = Offset(left, layout.chordTop)
        val size = Size(width, height)
        if (current) {
            drawRoundRect(MusicDecodedColors.Chord, topLeft, size, radius)
        } else {
            drawRoundRect(MusicDecodedColors.ChordMuted, topLeft, size, radius)
            drawRoundRect(MusicDecodedColors.ChordMutedBorder, topLeft, size, radius, style = Stroke(border))
        }
        val maxWidth = (width - 2 * padding).toInt()
        if (maxWidth < TimelineMetrics.minLabelWidth.toPx()) continue
        val text = textMeasurer.measure(
            chordName(chord.label),
            if (current) activeStyle else idleStyle,
            overflow = TextOverflow.Clip,
            maxLines = 1,
            constraints = Constraints(maxWidth = maxWidth),
        )
        drawText(text, topLeft = Offset(left + padding, layout.chordTop + (height - text.size.height) / 2f))
    }
}

private fun DrawScope.drawLoop(viewport: TimelineViewport, loop: OpenEndRange<Double>, layout: TimelineLayout) {
    val left = viewport.x(loop.start)
    val right = viewport.x(loop.endExclusive)
    if (right < 0f || left > size.width) return
    drawRect(MusicDecodedColors.Chord.copy(alpha = 0.14f), Offset(left, layout.chordTop), Size(right - left, layout.waveBottom - layout.chordTop))
    val stroke = Sizes.hairline.toPx() * 2
    drawLine(MusicDecodedColors.Chord, Offset(left, layout.chordTop), Offset(left, layout.waveBottom), stroke)
    drawLine(MusicDecodedColors.Chord, Offset(right, layout.chordTop), Offset(right, layout.waveBottom), stroke)
}

/** 1.5dp primary line with a dot on top and a soft glow (DESIGN.md §3.7). */
private fun DrawScope.drawPlayhead(viewport: TimelineViewport, layout: TimelineLayout) {
    val x = viewport.center
    val top = layout.chordTop - TimelineMetrics.playheadDot.toPx()
    drawLine(MusicDecodedColors.Primary.copy(alpha = 0.22f), Offset(x, top), Offset(x, layout.waveBottom), TimelineMetrics.playheadGlow.toPx(), StrokeCap.Round)
    drawLine(MusicDecodedColors.Primary, Offset(x, top), Offset(x, layout.waveBottom), TimelineMetrics.playhead.toPx())
    drawCircle(MusicDecodedColors.Primary, TimelineMetrics.playheadDot.toPx(), Offset(x, top))
}

@Preview(widthDp = 400)
@Composable
private fun ChordTimelinePreview() = PreviewSurface {
    val analysis = rememberPreviewAnalysis() ?: return@PreviewSurface
    var position by remember { mutableDoubleStateOf(16.0) }
    ChordTimeline(
        analysis = analysis,
        position = { position },
        chordName = { chordDisplayName(it, key = analysis.key) },
        onSeek = { position = it },
        loop = 12.841..<18.947,
    )
}
