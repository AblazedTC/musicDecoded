package com.musicdecoded.mobile.core

/**
 * Loads the real files from the repo's `contracts/` folder (wired in as a test resource dir in
 * `app/build.gradle.kts`), so the tests always run against the shared contracts, never a copy.
 */
object Contracts {
    fun text(path: String): String =
        checkNotNull(Contracts::class.java.getResource("/$path")) { "contracts/$path not on the test classpath" }
            .readText()

    fun sampleAnalysis(): Analysis = Analysis.decode(text("fixtures/sample.analysis.json"))

    fun voicings(): ChordVoicings = ChordVoicings.decode(text("chord-voicings.json"))
}

/**
 * A small hand-made analysis with runs of `N`, for edge cases the fixture doesn't have.
 *
 *     chords: A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10
 *     beats:  every 1 s from 0 to 9, downbeat every 4 (0, 4, 8)
 */
object SyntheticAnalysis {
    fun make(): Analysis = Analysis(
        schemaVersion = 1,
        meta = Analysis.Meta(
            trackId = "synthetic",
            source = Source.FILE,
            durationSec = 10.0,
            analyzedAt = "2026-01-01T00:00:00Z",
            analyzerVersion = "test",
        ),
        key = null,
        bpm = 60.0,
        timeSignature = Analysis.TimeSignature(4, 4),
        beats = (0 until 10).map { Analysis.Beat(it.toDouble(), it % 4 == 0) },
        chords = listOf(
            Analysis.Chord(0.0, 2.0, "A:maj"),
            Analysis.Chord(2.0, 4.0, "N"),
            Analysis.Chord(4.0, 5.0, "N"),
            Analysis.Chord(5.0, 8.0, "D:maj"),
            Analysis.Chord(8.0, 10.0, "N"),
        ),
    )
}
