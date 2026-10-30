package com.musicdecoded.mobile.player

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.CustomAccessibilityAction
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.customActions
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.core.isNoChord
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.theme.ArchivoBlack
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.SpaceGrotesk
import com.musicdecoded.mobile.ui.theme.Spacing
import androidx.compose.ui.text.font.FontWeight
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Fixed drawing sizes of the timeline (component sizes, not spacing). */
private object TimelineMetrics {
    val timeLabels = 16.dp
    val labelSpacing = 56.dp
    val minWaveform = 40.dp
    val barMaxWidth = 3.dp
    val playhead = 2.dp
    val chordName = 20.sp
    val numeral = 15.sp
}

private enum class Gesture { NONE, SCRUB, LOOP, ZOOM, IGNORE }

/**
 * The player timeline (DESIGN.md §3.7 item 3), edge to edge: a time-label row, chord-label boxes
 * (name + Roman numeral) at each chord change in up to 2 staggered rows with a tick down to the
 * waveform, the waveform with downbeat gridlines, and a fixed center playhead. Content scrolls under
 * the playhead. Drag the waveform to scrub, drag the label area to select a loop (snapped to beats),
 * tap a label to seek to its chord, pinch to zoom (60–400 dp/s). Fills the height it's given.
 *
 * [position] is read in the draw phase only, so a 60 Hz clock redraws without recomposing;
 * [currentIndex] (the current chord) only changes at chord changes and re-runs the label layout.
 */
@Composable
fun ChordTimeline(
    analysis: Analysis,
    position: () -> Double,
    chords: ChordDisplay,
    currentIndex: Int?,
    onSeek: (Double) -> Unit,
    modifier: Modifier = Modifier,
    loop: OpenEndRange<Double>? = null,
    onLoopSelected: (OpenEndRange<Double>) -> Unit = {},
) {
    var zoom by rememberSaveable { mutableFloatStateOf(DEFAULT_ZOOM_DP_PER_SEC) }
    var loopDraft by remember { mutableStateOf<OpenEndRange<Double>?>(null) }
    val currentPosition by rememberUpdatedState(position)
    val currentAnalysis by rememberUpdatedState(analysis)
    val seek by rememberUpdatedState(onSeek)
    val selectLoop by rememberUpdatedState(onLoopSelected)
    val localDensity = LocalDensity.current

    val textMeasurer = rememberTextMeasurer(cacheSize = 64)
    val timeStyle = MaterialTheme.typography.labelSmall.copy(color = MusicDecodedColors.TextMuted, fontFeatureSettings = "tnum")
    val nameStyle = TextStyle(fontFamily = ArchivoBlack, fontSize = TimelineMetrics.chordName, color = MusicDecodedColors.Text)
    val numeralStyle = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = TimelineMetrics.numeral, color = MusicDecodedColors.Text)

    // Measured once per chord set (names repeat, so cache by text); colors are applied when drawing.
    val boxes = remember(analysis, chords, localDensity, textMeasurer) {
        val cache = HashMap<String, TextLayoutResult>()
        fun measure(text: String, style: TextStyle) = cache.getOrPut(text + '\u0000' + style.fontSize) { textMeasurer.measure(text, style) }
        val hasNumerals = analysis.key != null
        val sample = measure("A", nameStyle).size.height + if (hasNumerals) measure("I", numeralStyle).size.height else 0
        with(localDensity) {
            val pad = Spacing.xxs.toPx() + Borders.separator.toPx()
            LabelBoxes(
                height = sample + 2 * pad,
                pad = pad,
                items = analysis.chords.map { chord ->
                    if (isNoChord(chord.label)) return@map null
                    val name = measure(chords.name(chord.label), nameStyle)
                    val numeral = chords.numeral(chord.label)?.let { measure(it, numeralStyle) }
                    LabelText(name, numeral, max(name.size.width, numeral?.size?.width ?: 0) + 2 * pad)
                },
            )
        }
    }
    val pxPerSec = zoom * localDensity.density
    val labels = remember(boxes, pxPerSec, currentIndex) {
        layoutChordLabels(analysis.chords, pxPerSec, width = { boxes.items[it]?.width ?: 0f }, current = currentIndex, gap = with(localDensity) { Spacing.xxs.toPx() })
    }
    val currentLabels by rememberUpdatedState(labels)
    val currentBoxes by rememberUpdatedState(boxes)
    val latestIndex by rememberUpdatedState(currentIndex)

    Canvas(
        modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Canvas)
            .clipToBounds()
            // Canvas content is invisible to TalkBack: describe it, and offer zoom as actions.
            .semantics {
                contentDescription = "Chord timeline. Pinch to zoom, drag to scrub."
                stateDescription = timelineSummary(analysis, chords, currentIndex)
                customActions = listOf(
                    CustomAccessibilityAction("Zoom in") { zoom = clampZoom(zoom * ZOOM_STEP); true },
                    CustomAccessibilityAction("Zoom out") { zoom = clampZoom(zoom / ZOOM_STEP); true },
                )
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    val pps = zoom * density
                    val startViewport = TimelineViewport(size.width.toFloat(), currentPosition(), pps)
                    val bands = TimelineBands(this, size.height.toFloat(), currentBoxes.height)
                    val onLabels = down.position.y < bands.labelsBottom
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
                                // Vertical drags belong to the screen (it scrolls at the largest text sizes).
                                gesture = when {
                                    abs(delta.y) > abs(delta.x) -> Gesture.IGNORE
                                    onLabels -> Gesture.LOOP
                                    else -> Gesture.SCRUB
                                }
                            }
                            val duration = currentAnalysis.meta.durationSec
                            when (gesture) {
                                Gesture.SCRUB -> seek((startViewport.playhead - delta.x / pps).coerceIn(0.0, duration))
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
                        Gesture.NONE -> if (onLabels) {
                            // Tap a label → seek to its chord.
                            val offset = startViewport.center - (startViewport.playhead * pps).toFloat()
                            val gap = Spacing.xxs.toPx()
                            currentLabels.indices.lastOrNull { i ->
                                val label = currentLabels[i]
                                val left = labelScreenLeft(currentLabels, i, offset, latestIndex, gap)
                                val top = bands.rowTop(label.row)
                                down.position.x in left..(left + label.width) &&
                                    down.position.y in top..(top + currentBoxes.height)
                            }?.let { seek(currentAnalysis.chords[currentLabels[it].index].start) }
                        }
                        Gesture.LOOP -> loopDraft?.let(selectLoop)
                        else -> Unit
                    }
                    loopDraft = null
                }
            },
    ) {
        val viewport = TimelineViewport(size.width, currentPosition(), pxPerSec)
        val bands = TimelineBands(this, size.height, boxes.height)
        drawTimeLabels(viewport, analysis, bands, textMeasurer, timeStyle)
        drawDownbeats(viewport, analysis, bands)
        drawWaveform(viewport, analysis, bands)
        (loopDraft ?: loop)?.let { drawLoop(viewport, it, bands) }
        drawChordLabels(viewport, analysis, bands, boxes, labels, currentIndex)
        drawPlayhead(viewport, bands)
    }
}

/** A chord's measured label: name, optional numeral, and the box width (px). */
private class LabelText(val name: TextLayoutResult, val numeral: TextLayoutResult?, val width: Float)

/** Every chord's label (`null` for `N`/`X`), with the shared box [height] and inner [pad] (px). */
private class LabelBoxes(val height: Float, val pad: Float, val items: List<LabelText?>)

/** Screen x of `labels[i]` given the content→screen [offset]; the current label sticks (see [stickyLabelLeft]). */
private fun labelScreenLeft(labels: List<PlacedLabel>, i: Int, offset: Float, currentIndex: Int?, gap: Float): Float {
    val label = labels[i]
    val left = label.left + offset
    if (label.index != currentIndex) return left
    val next = (i + 1 until labels.size).firstOrNull { labels[it].row == label.row }?.let { labels[it].left + offset }
    return stickyLabelLeft(left, label.width, next, gap)
}

/** "Current chord E, III. Next A, VI." for TalkBack. */
private fun timelineSummary(analysis: Analysis, chords: ChordDisplay, currentIndex: Int?): String {
    fun describe(label: String) = listOfNotNull(chords.name(label), chords.numeral(label)).joinToString(", ")
    val current = currentIndex?.let { analysis.chords[it].label }?.takeUnless(::isNoChord)
    val next = analysis.chords.drop((currentIndex ?: -1) + 1).firstOrNull { !isNoChord(it.label) }?.label
    return listOfNotNull(
        current?.let { "Current chord ${describe(it)}." } ?: "No chord.",
        next?.let { "Next ${describe(it)}." },
    ).joinToString(" ")
}

/** A band that holds text: [height] at 100% font size, growing with the user's font scale. */
private fun Density.scaled(height: Dp): Float = max(height.toPx(), height.value.sp.toPx())

/** One accessibility zoom step (×1.5 / ÷1.5). */
private const val ZOOM_STEP = 1.5f

/**
 * Vertical bands of the canvas, in px: time labels, 2 label rows, then the waveform down to the
 * bottom. When the height is short the waveform keeps at least [TimelineMetrics.minWaveform] and the
 * lower label row overlaps its top.
 */
private class TimelineBands(density: Density, height: Float, boxHeight: Float) {
    val timeBottom: Float
    private val rowsTop: Float
    private val rowStep: Float
    val labelsBottom: Float
    val waveTop: Float
    val waveBottom: Float

    init {
        with(density) {
            val gap = Spacing.xxs.toPx()
            timeBottom = scaled(TimelineMetrics.timeLabels)
            rowsTop = timeBottom + gap
            rowStep = boxHeight + gap
            labelsBottom = rowsTop + CHORD_LABEL_ROWS * rowStep
            waveBottom = height - gap
            waveTop = min(labelsBottom, waveBottom - TimelineMetrics.minWaveform.toPx())
        }
    }

    fun rowTop(row: Int): Float = rowsTop + row * rowStep
}

private fun DrawScope.drawTimeLabels(viewport: TimelineViewport, analysis: Analysis, bands: TimelineBands, textMeasurer: TextMeasurer, style: TextStyle) {
    val step = timeLabelStep(viewport.pxPerSec, TimelineMetrics.labelSpacing.toPx())
    var second = (ceil(max(0.0, viewport.startTime) / step) * step).toInt()
    val last = min(viewport.endTime, analysis.meta.durationSec)
    while (second <= last) {
        val text = textMeasurer.measure(formatDuration(second.toDouble()), style)
        val x = viewport.x(second.toDouble())
        drawLine(MusicDecodedColors.TextMuted, Offset(x, Spacing.xxs.toPx()), Offset(x, bands.timeBottom), Sizes.hairline.toPx())
        drawText(text, topLeft = Offset(x + Spacing.xxs.toPx(), (bands.timeBottom - text.size.height) / 2f))
        second += step
    }
}

private fun DrawScope.drawDownbeats(viewport: TimelineViewport, analysis: Analysis, bands: TimelineBands) {
    val color = MusicDecodedColors.Line.copy(alpha = 0.5f)
    val stroke = Sizes.hairline.toPx()
    for (i in visibleBeatIndices(analysis.beats, viewport.startTime, viewport.endTime)) {
        val beat = analysis.beats[i]
        if (!beat.downbeat) continue
        val x = viewport.x(beat.t)
        drawLine(color, Offset(x, bands.waveTop), Offset(x, bands.waveBottom), stroke)
    }
}

/** Waveform bars from `waveformPeaks` (played = primary), or beat ticks when there are no peaks. */
private fun DrawScope.drawWaveform(viewport: TimelineViewport, analysis: Analysis, bands: TimelineBands) {
    val middle = (bands.waveTop + bands.waveBottom) / 2f
    val height = bands.waveBottom - bands.waveTop
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

/**
 * A 1dp tick at every visible chord change (dropped labels keep theirs), then the placed label boxes:
 * current = lime + ink text + small hard shadow, others = cardRaised + white text; 2dp black border.
 */
private fun DrawScope.drawChordLabels(
    viewport: TimelineViewport,
    analysis: Analysis,
    bands: TimelineBands,
    boxes: LabelBoxes,
    labels: List<PlacedLabel>,
    currentIndex: Int?,
) {
    val tick = Sizes.hairline.toPx()
    val placedTop = HashMap<Int, Float>(labels.size)
    labels.forEach { placedTop[it.index] = bands.rowTop(it.row) }
    for (i in visibleChordIndices(analysis.chords, viewport.startTime, viewport.endTime)) {
        val chord = analysis.chords[i]
        if (isNoChord(chord.label)) continue
        val x = viewport.x(chord.start)
        val top = placedTop[i]?.let { it + boxes.height } ?: bands.rowTop(0)
        val color = if (i == currentIndex) MusicDecodedColors.Chord else MusicDecodedColors.TextMuted
        drawLine(color, Offset(x, top), Offset(x, bands.waveBottom), tick)
    }

    val offset = viewport.center - (viewport.playhead * viewport.pxPerSec).toFloat()
    val border = Borders.separator.toPx()
    val shadow = Shadows.sm.toPx()
    val gap = Spacing.xxs.toPx()
    for (i in labels.indices) {
        val label = labels[i]
        val left = labelScreenLeft(labels, i, offset, currentIndex, gap)
        if (left > size.width || left + label.width + shadow < 0f) continue
        val text = boxes.items[label.index] ?: continue
        val current = label.index == currentIndex
        val top = bands.rowTop(label.row)
        val boxSize = Size(label.width, boxes.height)
        if (current) drawRect(MusicDecodedColors.Violet, Offset(left + shadow, top + shadow), boxSize)
        drawRect(if (current) MusicDecodedColors.Chord else MusicDecodedColors.SurfaceRaised, Offset(left, top), boxSize)
        drawRect(MusicDecodedColors.Border, Offset(left + border / 2, top + border / 2), Size(label.width - border, boxes.height - border), style = Stroke(border))
        val textColor = if (current) MusicDecodedColors.OnChord else MusicDecodedColors.Text
        val nameTop = top + boxes.pad
        drawText(text.name, color = textColor, topLeft = Offset(left + (label.width - text.name.size.width) / 2f, nameTop))
        text.numeral?.let {
            drawText(it, color = textColor, topLeft = Offset(left + (label.width - it.size.width) / 2f, nameTop + text.name.size.height))
        }
    }
}

private fun DrawScope.drawLoop(viewport: TimelineViewport, loop: OpenEndRange<Double>, bands: TimelineBands) {
    val left = viewport.x(loop.start)
    val right = viewport.x(loop.endExclusive)
    if (right < 0f || left > size.width) return
    val top = bands.timeBottom
    drawRect(MusicDecodedColors.Secondary.copy(alpha = 0.18f), Offset(left, top), Size(right - left, bands.waveBottom - top))
    val stroke = Sizes.hairline.toPx() * 2
    drawLine(MusicDecodedColors.Secondary, Offset(left, top), Offset(left, bands.waveBottom), stroke)
    drawLine(MusicDecodedColors.Secondary, Offset(right, top), Offset(right, bands.waveBottom), stroke)
}

/** 2dp lime line at the center, through the labels and the waveform (DESIGN.md §3.7). */
private fun DrawScope.drawPlayhead(viewport: TimelineViewport, bands: TimelineBands) {
    val x = viewport.center
    drawLine(MusicDecodedColors.Primary, Offset(x, bands.timeBottom), Offset(x, bands.waveBottom), TimelineMetrics.playhead.toPx())
}

@Preview(widthDp = 400)
@Composable
private fun ChordTimelinePreview() = PreviewSurface {
    val analysis = rememberPreviewAnalysis() ?: return@PreviewSurface
    var position by remember { mutableDoubleStateOf(16.0) }
    ChordTimeline(
        analysis = analysis,
        position = { position },
        chords = ChordDisplay(analysis.key, 0, basic = false, voicings = null),
        currentIndex = currentState(position, analysis).chordIndex,
        onSeek = { position = it },
        modifier = Modifier.height(200.dp),
        loop = 12.841..<18.947,
    )
}
