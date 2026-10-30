/// One chord box in the Chord sheet view (DESIGN.md §3.7.1).
public struct ChordSheetEntry: Sendable, Equatable {
    public let label: String
    /// The `analysis.chords` indices this box stands for (a chord repeated within a bar is shown once).
    public let chordIndices: [Int]
    /// Where a tap seeks: the chord's start, or the bar's start when it carries over.
    public let time: Double

    public init(label: String, chordIndices: [Int], time: Double) {
        self.label = label
        self.chordIndices = chordIndices
        self.time = time
    }
}

/// One bar of the sheet; `entries` is empty for a bar with no chord.
public struct ChordSheetBar: Sendable, Equatable {
    /// 1-based.
    public let number: Int
    public let start: Double
    public let end: Double
    public let entries: [ChordSheetEntry]

    public init(number: Int, start: Double, end: Double, entries: [ChordSheetEntry]) {
        self.number = number
        self.start = start
        self.end = end
        self.entries = entries
    }
}

/// Chord-sheet layout (DESIGN.md §3.7.1): the progression by bar, 4 bars per line, repeated
/// chords within a bar shown once.
public enum ChordSheet {
    public static let barsPerLine = 4
    /// A chord must overlap a bar by more than this (seconds) to show in it, so a chord ending
    /// a hair after the downbeat isn't repeated in the next bar.
    static let overlapTolerance = 0.1

    /// The bars, from the beat grid's bars. Without beats, each real chord is its own bar.
    public static func bars(for analysis: Analysis) -> [ChordSheetBar] {
        let chords = analysis.chords
        let gridBars = BeatGrid.bars(for: analysis)
        guard gridBars.isEmpty == false else {
            let real = chords.indices.filter { isNoChord(chords[$0].label) == false }
            return real.enumerated().map { number, index in
                let chord = chords[index]
                return ChordSheetBar(
                    number: number + 1,
                    start: chord.start,
                    end: chord.end,
                    entries: [ChordSheetEntry(label: chord.label, chordIndices: [index], time: chord.start)]
                )
            }
        }
        return gridBars.map { bar in
            ChordSheetBar(number: bar.number, start: bar.start, end: bar.end, entries: entries(chords, from: bar.start, to: bar.end))
        }
    }

    /// The real chords overlapping `from..<to`, consecutive repeats merged.
    static func entries(_ chords: [Analysis.Chord], from start: Double, to end: Double) -> [ChordSheetEntry] {
        let tolerance = min(overlapTolerance, (end - start) / 4)
        var entries: [ChordSheetEntry] = []
        var index = max(upperBound(chords, start, \.start) - 1, 0)
        while index < chords.count, chords[index].start < end - tolerance {
            let chord = chords[index]
            if isNoChord(chord.label) == false, chord.end > start + tolerance {
                if let last = entries.last, last.label == chord.label {
                    entries[entries.count - 1] = ChordSheetEntry(label: last.label, chordIndices: last.chordIndices + [index], time: last.time)
                } else {
                    entries.append(ChordSheetEntry(label: chord.label, chordIndices: [index], time: max(chord.start, start)))
                }
            }
            index += 1
        }
        return entries
    }

    /// `bars` in lines of `perLine`.
    public static func lines(_ bars: [ChordSheetBar], perLine: Int = barsPerLine) -> [[ChordSheetBar]] {
        let perLine = max(perLine, 1)
        return stride(from: 0, to: bars.count, by: perLine).map { Array(bars[$0..<min($0 + perLine, bars.count)]) }
    }

    /// The bar index at `t` (the last bar keeps `t` past its end), or `nil` before the first bar.
    public static func barIndex(at t: Double, in bars: [ChordSheetBar]) -> Int? {
        let index = upperBound(bars, t, \.start) - 1
        guard bars.indices.contains(index) else { return nil }
        return t < bars[index].end || index == bars.count - 1 ? index : nil
    }

    /// The current box: the entry of the bar at `t` that stands for chord `chordIndex`.
    public static func current(at t: Double, chordIndex: Int?, in bars: [ChordSheetBar]) -> (bar: Int, entry: Int)? {
        guard let chordIndex, let bar = barIndex(at: t, in: bars),
              let entry = bars[bar].entries.firstIndex(where: { $0.chordIndices.contains(chordIndex) }) else {
            return nil
        }
        return (bar, entry)
    }
}
