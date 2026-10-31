package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Contracts
import com.musicdecoded.mobile.core.SyntheticAnalysis
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PlayerViewsMathTest {
    private fun analysis(beats: List<Analysis.Beat>, chords: List<Analysis.Chord>, numerator: Int = 4) =
        SyntheticAnalysis.make().copy(beats = beats, chords = chords, timeSignature = Analysis.TimeSignature(numerator, 4))

    @Test fun beatsGroupIntoBarsAtDownbeatsWithPadding() {
        val grid = beatGrid(SyntheticAnalysis.make())
        assertEquals(listOf(1, 2, 3), grid.map { it.number })
        assertEquals(listOf(0, 1, 2, 3), grid[0].cells.map { it.beat })
        assertEquals(listOf(8, 9, -1, -1), grid[2].cells.map { it.beat })
        assertEquals(4.0, grid[1].start, 0.0)
        assertEquals(8.0, grid[1].end, 0.0)
        assertEquals(10.0, grid[2].end, 0.0)
    }

    @Test fun aPickupBarIsPaddedAtTheStart() {
        val beats = (0 until 8).map { Analysis.Beat(it.toDouble(), (it - 2) % 3 == 0 && it >= 2) }
        val grid = beatGrid(analysis(beats, listOf(Analysis.Chord(0.0, 8.0, "C:maj")), numerator = 3))
        assertEquals(listOf(-1, 0, 1), grid[0].cells.map { it.beat })
        assertEquals(listOf(2, 3, 4), grid[1].cells.map { it.beat })
        assertEquals(listOf(5, 6, 7), grid[2].cells.map { it.beat })
    }

    @Test fun withoutDownbeatsBarsAreEveryNumeratorBeats() {
        val beats = (0 until 6).map { Analysis.Beat(it.toDouble(), false) }
        val grid = beatGrid(analysis(beats, emptyList(), numerator = 4))
        assertEquals(listOf(listOf(0, 1, 2, 3), listOf(4, 5, -1, -1)), grid.map { bar -> bar.cells.map { it.beat } })
    }

    @Test fun chordStartsMapToTheNearestBeatAndContinue() {
        val cells = beatGrid(SyntheticAnalysis.make()).flatMap { it.cells }.filter { it.beat >= 0 }
        // A:maj 0–2 · N 2–5 · D:maj 5–8 · N 8–10
        assertEquals(listOf(0, 0, null, null, null, 3, 3, 3, null, null), cells.map { it.chord })
        assertEquals(listOf(0, 5), cells.filter { it.starts }.map { it.beat })

        val early = analysis(SyntheticAnalysis.make().beats, listOf(Analysis.Chord(0.0, 2.96, "A:maj"), Analysis.Chord(2.96, 10.0, "E:maj")))
        val starts = beatGrid(early).flatMap { it.cells }.filter { it.starts }.map { it.beat to it.chord }
        assertEquals(listOf(0 to 0, 3 to 1), starts) // 2.96 s snaps to beat 3
    }

    @Test fun sheetDedupesRepeatedChordsWithinABar() {
        val chords = listOf(
            Analysis.Chord(0.0, 1.0, "A:maj"),
            Analysis.Chord(1.0, 2.0, "A:maj"),
            Analysis.Chord(2.0, 5.0, "D:maj"),
            Analysis.Chord(5.0, 10.0, "D:maj"),
        )
        val sheet = chordSheet(analysis(SyntheticAnalysis.make().beats, chords))
        assertEquals(listOf("A:maj", "D:maj"), sheet[0].chords.map { chords[it.chord].label })
        assertEquals(0..1, sheet[0].chords[0].beats)
        assertEquals(2..3, sheet[0].chords[1].beats)
        // D carries into bar 2: one box seeking to the bar's first beat.
        assertEquals(1, sheet[1].chords.size)
        assertEquals(4.0, sheet[1].chords[0].seekTo, 0.0)
    }

    @Test fun sheetSkipsNoChordAndFallsBackWithoutBeats() {
        val sheet = chordSheet(SyntheticAnalysis.make())
        assertEquals(listOf(1, 1, 0), sheet.map { it.chords.size })
        assertEquals(5.0, sheet[1].chords[0].seekTo, 0.0)

        val noBeats = chordSheet(SyntheticAnalysis.make().copy(beats = emptyList()))
        assertEquals(listOf(0, 3), noBeats.map { it.chords.single().chord })
    }

    @Test fun beatLookups() {
        val beats = SyntheticAnalysis.make().beats
        assertEquals(-1, beatAt(-0.5, beats))
        assertEquals(3, beatAt(3.99, beats))
        assertEquals(4, nearestBeat(3.6, beats))
        assertEquals(1, barOfBeat(beatGrid(SyntheticAnalysis.make()), 5))
    }

    @Test fun fixtureGridIsWellFormed() {
        val analysis = Contracts.sampleAnalysis()
        val grid = beatGrid(analysis)
        val perBar = analysis.timeSignature?.numerator ?: 4
        assertTrue(grid.isNotEmpty())
        assertTrue(grid.all { it.cells.size == perBar })
        assertEquals(analysis.beats.indices.toList(), grid.flatMap { it.cells }.map { it.beat }.filter { it >= 0 })
    }
}
