package com.musicdecoded.mobile.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Fixture landmarks used below:
 *     chords[0] N 0–0.72 · [1] A:maj 0.72–1.138 · [2] D:maj 1.138–2.345 · [3] D:maj7 2.345–5.596
 *     … [92] C:7 256.209–259.738 · [93] N 259.738–263.407
 *     beats 0.337 (downbeat), 0.789, 1.242, 1.672, 2.101 (downbeat), 2.531, … (581 total)
 */
class TimelineTest {
    private val fixture = Contracts.sampleAnalysis()
    private val synthetic = SyntheticAnalysis.make()

    // currentState

    @Test fun beforeEverything() {
        assertEquals(PlaybackState(null, null, null), currentState(-0.5, fixture))
    }

    @Test fun insideLeadingNoChordBeforeFirstBeat() {
        assertEquals(PlaybackState(0, null, null), currentState(0.1, fixture))
    }

    @Test fun exactlyOnAChordBoundaryBelongsToTheNewChord() {
        assertEquals(1, currentState(0.72, fixture).chordIndex)
        assertEquals(2, currentState(1.138, fixture).chordIndex)
        assertEquals(0, currentState(0.0, fixture).chordIndex)
    }

    @Test fun exactlyOnADownbeatStartsTheNextBar() {
        assertEquals(PlaybackState(2, 4, 1), currentState(2.101, fixture))
        assertEquals(0, currentState(2.1, fixture).bar)
        assertEquals(0, currentState(0.337, fixture).beatIndex)
    }

    @Test fun lastChordAndAfterTheEnd() {
        assertEquals(93, currentState(263.0, fixture).chordIndex)
        assertNull("end is exclusive", currentState(263.407, fixture).chordIndex)
        assertNull(currentState(300.0, fixture).chordIndex)
        assertEquals(fixture.beats.size - 1, currentState(300.0, fixture).beatIndex)
    }

    @Test fun barCountsDownbeats() {
        val downbeats = fixture.beats.filter { it.downbeat }
        assertEquals(downbeats.size - 1, currentState(downbeats.last().t, fixture).bar)
    }

    // upNext

    @Test fun upNextSkipsLeadingNoChord() {
        assertEquals(UpNext(1, 1), upNext(-1.0, fixture))
        assertEquals(UpNext(1, 1), upNext(0.1, fixture))
    }

    @Test fun upNextCountsBeatsUntilTheChord() {
        // A:maj → D:maj at 1.138; no beat in (0.8, 1.138].
        assertEquals(UpNext(2, 0), upNext(0.8, fixture))
        // D:maj → D:maj7 at 2.345; beats 1.242, 1.672, 2.101.
        assertEquals(UpNext(3, 3), upNext(1.2, fixture))
    }

    @Test fun upNextSkipsRunsOfNoChord() {
        // A:maj, then N, N, then D:maj at 5. Beats 1, 2, 3, 4, 5 are in (0.5, 5].
        assertEquals(UpNext(3, 5), upNext(0.5, synthetic))
        assertEquals(UpNext(3, 2), upNext(3.0, synthetic))
    }

    @Test fun upNextIsNullWhenOnlyNoChordRemains() {
        assertNull(upNext(257.0, fixture))
        assertNull(upNext(6.0, synthetic))
        assertNull(upNext(500.0, fixture))
    }

    // previous / next chord

    @Test fun previousRestartsCurrentChordAfterOneSecond() {
        assertEquals(2.345, previousChordStart(3.5, fixture))
        assertEquals(5.0, previousChordStart(6.5, synthetic))
    }

    @Test fun previousGoesBackWithinTheFirstSecond() {
        assertEquals(0.72, previousChordStart(1.5, fixture))
        // Exactly 1 s in is not "more than 1 s", so it goes to the previous chord.
        assertEquals(0.0, previousChordStart(6.0, synthetic))
    }

    @Test fun previousSkipsNoChord() {
        assertEquals("skips N 4–5 and N 2–4", 0.0, previousChordStart(5.5, synthetic))
        assertEquals("inside N, go to the chord before it", 0.0, previousChordStart(3.0, synthetic))
        assertEquals(5.0, previousChordStart(9.5, synthetic))
        assertEquals(256.209, previousChordStart(261.0, fixture))
    }

    @Test fun previousIsNullAtTheStart() {
        assertNull(previousChordStart(-1.0, fixture))
        assertNull("only N before", previousChordStart(0.1, fixture))
        assertNull("first chord, under 1 s in", previousChordStart(0.9, fixture))
        assertNull(previousChordStart(0.5, synthetic))
    }

    @Test fun nextSkipsNoChord() {
        assertEquals(0.72, nextChordStart(-1.0, fixture))
        assertEquals(0.72, nextChordStart(0.1, fixture))
        assertEquals(1.138, nextChordStart(0.72, fixture))
        assertEquals(5.0, nextChordStart(1.0, synthetic))
    }

    @Test fun nextIsNullAfterTheLastChord() {
        assertNull(nextChordStart(257.0, fixture))
        assertNull(nextChordStart(5.0, synthetic))
    }

    // barRange

    @Test fun barRangeIsDownbeatToDownbeat() {
        assertEquals(0.337..<2.101, barRange(1.0, fixture))
        assertEquals(2.101..<3.855, barRange(2.101, fixture))
        assertEquals(4.0..<8.0, barRange(5.0, synthetic))
    }

    @Test fun barRangeBeforeTheFirstDownbeatIsThePickup() {
        assertEquals(0.0..<0.337, barRange(0.1, fixture))
    }

    @Test fun barRangeAfterTheLastDownbeatEndsAtTheDuration() {
        val last = fixture.beats.last { it.downbeat }
        assertEquals(last.t..<263.407, barRange(263.0, fixture))
        assertEquals(8.0..<10.0, barRange(9.0, synthetic))
    }

    @Test fun barRangeIsNullOutsideThePlayableRange() {
        assertNull("exactly at durationSec (exclusive end)", barRange(263.407, fixture))
        assertNull(barRange(300.0, fixture))
        assertNull(barRange(-1.0, fixture))
    }

    @Test fun barRangeIsNullWithoutDownbeats() {
        val analysis = synthetic.copy(beats = synthetic.beats.map { it.copy(downbeat = false) })
        assertNull(barRange(1.0, analysis))
    }
}
