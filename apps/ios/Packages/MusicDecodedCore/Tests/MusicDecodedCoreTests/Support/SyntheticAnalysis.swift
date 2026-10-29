@testable import MusicDecodedCore

/// A small hand-made analysis with runs of `N`, for edge cases the fixture doesn't have.
///
///     chords: A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10
///     beats:  every 1 s from 0 to 9, downbeat every 4 (0, 4, 8)
enum SyntheticAnalysis {
    static func make() -> Analysis {
        Analysis(
            schemaVersion: 1,
            meta: Analysis.Meta(
                trackId: "synthetic",
                source: .file,
                durationSec: 10,
                analyzedAt: "2026-01-01T00:00:00Z",
                analyzerVersion: "test"
            ),
            key: nil,
            bpm: 60,
            timeSignature: Analysis.TimeSignature(numerator: 4, denominator: 4),
            beats: (0..<10).map { Analysis.Beat(t: Double($0), downbeat: $0 % 4 == 0) },
            chords: [
                Analysis.Chord(start: 0, end: 2, label: "A:maj"),
                Analysis.Chord(start: 2, end: 4, label: "N"),
                Analysis.Chord(start: 4, end: 5, label: "N"),
                Analysis.Chord(start: 5, end: 8, label: "D:maj"),
                Analysis.Chord(start: 8, end: 10, label: "N"),
            ]
        )
    }
}
