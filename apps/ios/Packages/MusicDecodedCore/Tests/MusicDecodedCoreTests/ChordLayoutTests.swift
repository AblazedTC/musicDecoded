import Testing
@testable import MusicDecodedCore

/// Beat grid and chord sheet layout (DESIGN.md §3.7.1), on `SyntheticAnalysis`:
///     chords: A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10; beats every 1 s, downbeats 0, 4, 8.
struct ChordLayoutTests {
    let synthetic = SyntheticAnalysis.make()

    private func analysis(chords: [Analysis.Chord], beats: [Analysis.Beat], numerator: Int? = 4) -> Analysis {
        Analysis(
            schemaVersion: 1,
            meta: Analysis.Meta(
                trackId: "t",
                source: .file,
                durationSec: 8,
                analyzedAt: "2026-01-01T00:00:00Z",
                analyzerVersion: "test"
            ),
            key: nil,
            bpm: 60,
            timeSignature: numerator.map { Analysis.TimeSignature(numerator: $0, denominator: 4) },
            beats: beats,
            chords: chords
        )
    }

    private func beats(_ downbeats: [Bool]) -> [Analysis.Beat] {
        downbeats.indices.map { Analysis.Beat(t: Double($0), downbeat: downbeats[$0]) }
    }

    /// `[bar, entry]` of the current chord box, or `[]`.
    private func current(_ t: Double, _ chordIndex: Int?, _ bars: [ChordSheetBar]) -> [Int] {
        guard let found = ChordSheet.current(at: t, chordIndex: chordIndex, in: bars) else { return [] }
        return [found.bar, found.entry]
    }

    // MARK: Beat grid

    @Test func groupsSplitAtDownbeatsAndAfterAFullBar() {
        let split = beats([true, false, false, false, false, false, true, false])
        #expect(BeatGrid.groups(split, perBar: 4) == [[0, 1, 2, 3], [4, 5], [6, 7]])
        #expect(BeatGrid.groups(beats([false, true, false]), perBar: 4) == [[0], [1, 2]])
        #expect(BeatGrid.groups([], perBar: 4).isEmpty)
    }

    @Test func barsCellsAndChordStarts() {
        let bars = BeatGrid.bars(for: synthetic)
        #expect(bars.map(\.number) == [1, 2, 3])
        #expect(bars.map(\.start) == [0, 4, 8])
        #expect(bars.map(\.end) == [4, 8, 10])
        let cells = bars.flatMap(\.cells)
        #expect(cells.count == 10)
        #expect(cells.map(\.chordIndex) == [0, nil, nil, nil, nil, 3, nil, nil, nil, nil])
        #expect(cells.map(\.continues) == [false, true, false, false, false, false, true, true, false, false])
    }

    @Test func chordStartSnapsToTheNearestBeat() {
        let grid = beats([true, false, false, false])
        let chords = [
            Analysis.Chord(start: 0, end: 1.9, label: "C:maj"),
            Analysis.Chord(start: 1.9, end: 2.4, label: "F:maj"),
            Analysis.Chord(start: 2.4, end: 4, label: "G:maj"),
        ]
        #expect(BeatGrid.chordStarts(chords, beats: grid) == [0: 0, 2: 2])
        #expect(BeatGrid.nearestBeat(to: -1, in: grid) == 0)
        #expect(BeatGrid.nearestBeat(to: 9, in: grid) == 3)
        #expect(BeatGrid.nearestBeat(to: 1.5, in: grid) == 1)
    }

    @Test func beatsPerBarComesFromTheTimeSignature() {
        #expect(BeatGrid.beatsPerBar(analysis(chords: [], beats: [], numerator: 3)) == 3)
        #expect(BeatGrid.beatsPerBar(analysis(chords: [], beats: [], numerator: nil)) == 4)
        #expect(BeatGrid.bars(for: analysis(chords: [], beats: [])).isEmpty)
    }

    @Test func barIndexOfBeat() {
        let bars = BeatGrid.bars(for: synthetic)
        #expect(BeatGrid.barIndex(ofBeat: 0, in: bars) == 0)
        #expect(BeatGrid.barIndex(ofBeat: 5, in: bars) == 1)
        #expect(BeatGrid.barIndex(ofBeat: 9, in: bars) == 2)
        #expect(BeatGrid.barIndex(ofBeat: 10, in: bars) == nil)
    }

    @Test func fixtureBarsCoverEveryBeat() throws {
        let fixture = try Contracts.sampleAnalysis()
        let bars = BeatGrid.bars(for: fixture)
        #expect(bars.flatMap(\.cells).map(\.beatIndex) == Array(fixture.beats.indices))
        #expect(bars.allSatisfy { $0.cells.count <= BeatGrid.beatsPerBar(fixture) })
    }

    // MARK: Chord sheet

    @Test func sheetBarsFromTheGrid() {
        let bars = ChordSheet.bars(for: synthetic)
        #expect(bars.map { $0.entries.map(\.label) } == [["A:maj"], ["D:maj"], []])
        #expect(bars[1].entries[0].time == 5)
        #expect(current(6, 3, bars) == [1, 0])
        #expect(current(3, 1, bars) == [])
    }

    @Test func repeatsInABarShowOnceAndCarryOversStartAtTheBar() {
        let chords = [
            Analysis.Chord(start: 0, end: 1, label: "C:maj"),
            Analysis.Chord(start: 1, end: 2, label: "C:maj"),
            Analysis.Chord(start: 2, end: 6, label: "G:maj"),
            Analysis.Chord(start: 6, end: 8, label: "C:maj"),
        ]
        let grid = beats([true, false, false, false, true, false, false, false])
        let bars = ChordSheet.bars(for: analysis(chords: chords, beats: grid))
        #expect(bars.count == 2)
        #expect(bars[0].entries == [
            ChordSheetEntry(label: "C:maj", chordIndices: [0, 1], time: 0),
            ChordSheetEntry(label: "G:maj", chordIndices: [2], time: 2),
        ])
        #expect(bars[1].entries == [
            ChordSheetEntry(label: "G:maj", chordIndices: [2], time: 4),
            ChordSheetEntry(label: "C:maj", chordIndices: [3], time: 6),
        ])
        #expect(current(1.5, 1, bars) == [0, 0])
        #expect(current(4.5, 2, bars) == [1, 0])
    }

    @Test func aHairOfOverlapDoesNotRepeatAChord() {
        let chords = [
            Analysis.Chord(start: 0, end: 4.05, label: "C:maj"),
            Analysis.Chord(start: 4.05, end: 8, label: "G:maj"),
        ]
        let grid = beats([true, false, false, false, true, false, false, false])
        let bars = ChordSheet.bars(for: analysis(chords: chords, beats: grid))
        #expect(bars.map { $0.entries.map(\.label) } == [["C:maj"], ["G:maj"]])
    }

    @Test func withoutBeatsEachChordIsABar() {
        let chords = [
            Analysis.Chord(start: 0, end: 1, label: "N"),
            Analysis.Chord(start: 1, end: 3, label: "C:maj"),
            Analysis.Chord(start: 3, end: 5, label: "G:maj"),
        ]
        let bars = ChordSheet.bars(for: analysis(chords: chords, beats: []))
        #expect(bars.map(\.number) == [1, 2])
        #expect(bars.map { $0.entries.map(\.chordIndices) } == [[[1]], [[2]]])
    }

    @Test func linesOfFourBars() {
        let bars = (1...9).map { ChordSheetBar(number: $0, start: Double($0), end: Double($0 + 1), entries: []) }
        #expect(ChordSheet.lines(bars).map { $0.map(\.number) } == [[1, 2, 3, 4], [5, 6, 7, 8], [9]])
        #expect(ChordSheet.lines([]).isEmpty)
    }

    @Test func barIndexAtTime() {
        let bars = ChordSheet.bars(for: synthetic)
        #expect(ChordSheet.barIndex(at: -1, in: bars) == nil)
        #expect(ChordSheet.barIndex(at: 0, in: bars) == 0)
        #expect(ChordSheet.barIndex(at: 7.9, in: bars) == 1)
        #expect(ChordSheet.barIndex(at: 12, in: bars) == 2)
    }
}
