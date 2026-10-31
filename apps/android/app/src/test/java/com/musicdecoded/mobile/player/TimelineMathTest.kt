package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Contracts
import com.musicdecoded.mobile.core.SyntheticAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TimelineMathTest {
    private val synthetic = SyntheticAnalysis.make()

    @Test fun viewportKeepsThePlayheadAtTheCenter() {
        val viewport = TimelineViewport(width = 400f, playhead = 10.0, pxPerSec = 100f)
        assertEquals(200f, viewport.x(10.0), 0.001f)
        assertEquals(300f, viewport.x(11.0), 0.001f)
        assertEquals(8.0, viewport.startTime, 1e-9)
        assertEquals(12.0, viewport.endTime, 1e-9)
        assertEquals(11.5, viewport.time(350f), 1e-9)
    }

    @Test fun zoomIsClampedTo60to400() {
        assertEquals(60f, clampZoom(10f))
        assertEquals(400f, clampZoom(1000f))
        assertEquals(120f, clampZoom(DEFAULT_ZOOM_DP_PER_SEC))
    }

    @Test fun visibleChordsAreOnlyThoseOverlappingTheWindow() {
        // A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10
        assertEquals(1..3, visibleChordIndices(synthetic.chords, 3.0, 6.0))
        assertEquals(0..0, visibleChordIndices(synthetic.chords, -5.0, 1.0))
        assertEquals(4..4, visibleChordIndices(synthetic.chords, 9.0, 20.0))
        assertTrue(visibleChordIndices(emptyList(), 0.0, 1.0).isEmpty())
    }

    @Test fun visibleBeatsAreInclusive() {
        assertEquals(2..5, visibleBeatIndices(synthetic.beats, 2.0, 5.0))
        assertTrue(visibleBeatIndices(synthetic.beats, -3.0, -1.0).isEmpty())
    }

    @Test fun visiblePeaksFollowFps() {
        val peaks = Analysis.WaveformPeaks(fps = 10.0, peaks = List(100) { 0.5 })
        assertEquals(20..40, visiblePeakIndices(peaks, 2.0, 4.0))
        assertEquals(0..5, visiblePeakIndices(peaks, -3.0, 0.5))
        assertEquals(90..99, visiblePeakIndices(peaks, 9.0, 30.0))
        assertTrue(visiblePeakIndices(peaks, -3.0, -1.0).isEmpty())
        assertTrue(visiblePeakIndices(peaks, 11.0, 12.0).isEmpty())
    }

    @Test fun sunnyDaysFixtureDrawsAFewHundredPeaksPerScreen() {
        val analysis = Analysis.decode(Contracts.text("fixtures/sunnydays.analysis.json"))
        val peaks = checkNotNull(analysis.waveformPeaks)
        // 400 px wide at 120 px/s ≈ 3.3 s visible → ~67 bars, not all 4976.
        val viewport = TimelineViewport(400f, 60.0, 120f)
        val visible = visiblePeakIndices(peaks, viewport.startTime, viewport.endTime)
        assertTrue(visible.count() in 60..70)
    }

    @Test fun snapsToTheNearestBeat() {
        assertEquals(3.0, snapToBeat(3.4, synthetic.beats), 1e-9)
        assertEquals(4.0, snapToBeat(3.6, synthetic.beats), 1e-9)
        assertEquals(0.0, snapToBeat(-2.0, synthetic.beats), 1e-9)
        assertEquals(9.0, snapToBeat(42.0, synthetic.beats), 1e-9)
        assertEquals(1.7, snapToBeat(1.7, emptyList()), 1e-9)
    }

    @Test fun draggedLoopSnapsInEitherDirection() {
        assertEquals(2.0..<6.0, snappedLoop(5.8, 2.2, synthetic))
        assertEquals(2.0..<6.0, snappedLoop(2.2, 5.8, synthetic))
    }

    @Test fun tinyDragExtendsToTheNextBeat() {
        assertEquals(3.0..<4.0, snappedLoop(3.1, 3.2, synthetic))
    }

    @Test fun dragPastTheLastBeatEndsAtTheDuration() {
        assertEquals(9.0..<10.0, snappedLoop(9.2, 9.3, synthetic))
    }

    @Test fun noBeatsStillGivesALoop() {
        val noBeats = synthetic.copy(beats = emptyList())
        assertEquals(2.5..<4.0, snappedLoop(2.5, 4.0, noBeats))
        assertNull(snappedLoop(-5.0, -1.0, noBeats.copy(meta = noBeats.meta.copy(durationSec = 0.0))))
    }

    /** One chord per start time (each lasting until the next), labels 50 px wide unless given. */
    private fun chordsAt(vararg starts: Double, label: String = "C:maj") =
        starts.mapIndexed { i, start -> Analysis.Chord(start, starts.getOrElse(i + 1) { start + 1 }, label) }

    private fun layout(chords: List<Analysis.Chord>, pxPerSec: Float = 100f, current: Int? = null, width: Float = 50f) =
        layoutChordLabels(chords, pxPerSec, width = { width }, current = current, gap = 4f)

    @Test fun labelsThatFitStayInTheTopRow() {
        val placed = layout(chordsAt(0.0, 1.0, 2.0))
        assertEquals(listOf(0, 0, 0), placed.map { it.row })
        assertEquals(listOf(0f, 100f, 200f), placed.map { it.left })
    }

    @Test fun anOverlappingLabelMovesToTheSecondRow() {
        // 0.3 s apart at 100 px/s = 30 px < 50 + 4.
        val placed = layout(chordsAt(0.0, 0.3, 1.0))
        assertEquals(listOf(0 to 0, 1 to 1, 2 to 0), placed.map { it.index to it.row })
    }

    @Test fun theGapCountsAsOverlap() {
        assertEquals(listOf(0, 1), layout(chordsAt(0.0, 0.52)).map { it.row }) // 52 < 54
        assertEquals(listOf(0, 0), layout(chordsAt(0.0, 0.54)).map { it.row })
    }

    @Test fun aLabelThatFitsNeitherRowIsDropped() {
        val placed = layout(chordsAt(0.0, 0.2, 0.4, 1.0))
        assertEquals(listOf(0, 1, 3), placed.map { it.index })
        assertTrue(placed.all { it.row < CHORD_LABEL_ROWS })
    }

    @Test fun theCurrentLabelIsNeverDropped() {
        val placed = layout(chordsAt(0.0, 0.2, 0.4, 1.0), current = 2)
        assertEquals(listOf(1, 2, 3), placed.map { it.index }) // replaced chord 0 (the row that ended first)
        assertEquals(0, placed.first { it.index == 2 }.row)
    }

    @Test fun laterLabelsNeverDisplaceTheCurrentOne() {
        val placed = layout(chordsAt(0.0, 0.2, 0.4, 0.45), current = 1)
        assertEquals(listOf(0, 1), placed.map { it.index })
    }

    @Test fun noChordGetsNoLabel() {
        assertEquals(listOf(0, 3), layout(synthetic.chords).map { it.index })
    }

    @Test fun zoomingInReLaysOutLabelsIntoOneRow() {
        val chords = chordsAt(0.0, 0.2, 0.4, 0.6)
        assertEquals(listOf(0, 1, 3), layout(chords, pxPerSec = 100f).map { it.index })
        val zoomedIn = layout(chords, pxPerSec = 400f) // 80 px apart
        assertEquals(listOf(0, 0, 0, 0), zoomedIn.map { it.row })
        assertEquals(listOf(0f, 80f, 160f, 240f), zoomedIn.map { it.left })
    }

    @Test fun currentLabelSticksToTheLeftEdgeUntilPushedOff() {
        assertEquals(120f, stickyLabelLeft(left = 120f, width = 50f, nextLeft = 300f, gap = 4f)) // on screen: anchored
        assertEquals(0f, stickyLabelLeft(left = -200f, width = 50f, nextLeft = 300f, gap = 4f)) // scrolled off: held
        assertEquals(-24f, stickyLabelLeft(left = -200f, width = 50f, nextLeft = 30f, gap = 4f)) // pushed by the next
        assertEquals(0f, stickyLabelLeft(left = -200f, width = 50f, nextLeft = null, gap = 4f))
    }

    @Test fun memoizedLayoutMatchesAFreshLayoutForEveryCurrentChord() {
        // The real song at the narrowest zoom (most collisions): base + withCurrentLabel == layout(current).
        val chords = Analysis.decode(Contracts.text("fixtures/sunnydays.analysis.json")).chords
        val relayout = { current: Int? -> layout(chords, pxPerSec = 60f * 2.625f, current = current, width = 120f) }
        val base = relayout(null)
        assertTrue(base.size < chords.size) // some labels are dropped, so both paths are exercised
        for (current in listOf<Int?>(null) + chords.indices) {
            assertEquals(relayout(current), withCurrentLabel(base, current, relayout))
        }
    }

    @Test fun withCurrentLabelReusesTheBaseWhenItHasTheCurrentLabel() {
        val chords = chordsAt(0.0, 0.2, 0.4, 1.0)
        val base = layout(chords)
        var relayouts = 0
        val relayout = { current: Int? -> relayouts++; layout(chords, current = current) }
        assertTrue(withCurrentLabel(base, 1, relayout) === base)
        assertTrue(withCurrentLabel(base, null, relayout) === base)
        assertEquals(0, relayouts)
        assertEquals(listOf(1, 2, 3), withCurrentLabel(base, 2, relayout).map { it.index })
        assertEquals(1, relayouts)
    }

    @Test fun bandNeedsOneRowOnlyWhenEveryLabelFitsInRowZero() {
        assertEquals(1, chordLabelRowsNeeded(emptyList()))
        assertEquals(1, chordLabelRowsNeeded(layout(chordsAt(0.0, 1.0, 2.0))))
        assertEquals(2, chordLabelRowsNeeded(layout(chordsAt(0.0, 0.3, 1.0))))
    }

    @Test fun bandRowsFollowZoomButNotTheCurrentChord() {
        val chords = chordsAt(0.0, 1.0, 2.0, 3.0)
        val rowsAt = { pxPerSec: Float -> chordLabelRowsNeeded(layout(chords, pxPerSec = pxPerSec, width = 50f)) }
        assertEquals(1, rowsAt(100f))
        assertEquals(2, rowsAt(40f))
        // One row: every label is already in the base, so no chord change can move anything.
        val base = layout(chords, pxPerSec = 100f, width = 50f)
        for (current in chords.indices) {
            assertTrue(withCurrentLabel(base, current) { layout(chords, pxPerSec = 100f, current = it, width = 50f) } === base)
        }
    }

    @Test fun labelRowsLooksUpEachChordsRow() {
        val rows = LabelRows(layout(chordsAt(0.0, 0.3, 1.0)), chordCount = 4)
        assertEquals(listOf(0, 1, 0, -1, -1), listOf(0, 1, 2, 3, 99).map(rows::rowOf))
    }

    @Test fun seekStepQuantizesThePosition() {
        assertEquals(0, seekStep(-1.0, 248f))
        assertEquals(500, seekStep(124.0, 248f))
        assertEquals(500, seekStep(124.1, 248f)) // under a step: same value, no recomposition
        assertEquals(1000, seekStep(300.0, 248f))
    }

    @Test fun labelStepKeepsLabelsApart() {
        assertEquals(1, timeLabelStep(pxPerSec = 120f, minSpacingPx = 56f))
        assertEquals(2, timeLabelStep(pxPerSec = 40f, minSpacingPx = 56f))
        assertEquals(60, timeLabelStep(pxPerSec = 0.1f, minSpacingPx = 56f))
    }
}
