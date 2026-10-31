/// One beat of the Beat grid view (DESIGN.md §3.7.1).
public struct BeatGridCell: Sendable, Equatable {
    /// Index into `analysis.beats`.
    public let beatIndex: Int
    /// The beat's time (tap = seek here).
    public let time: Double
    /// The chord (index into `analysis.chords`) that starts in this beat, if any.
    public let chordIndex: Int?
    /// A chord is sounding here but didn't start here (thin lime underline).
    public let continues: Bool

    public init(beatIndex: Int, time: Double, chordIndex: Int?, continues: Bool) {
        self.beatIndex = beatIndex
        self.time = time
        self.chordIndex = chordIndex
        self.continues = continues
    }
}

/// One row of the Beat grid: a bar of at most `beatsPerBar` cells.
public struct BeatGridBar: Sendable, Equatable {
    /// 1-based bar number shown on the left.
    public let number: Int
    public let start: Double
    public let end: Double
    public let cells: [BeatGridCell]

    public init(number: Int, start: Double, end: Double, cells: [BeatGridCell]) {
        self.number = number
        self.start = start
        self.end = end
        self.cells = cells
    }
}

/// Beat-grid layout (DESIGN.md §3.7.1): beats → bars, chord starts → beats.
public enum BeatGrid {
    /// Upper bound for odd time signatures, so a row always fits the screen.
    public static let maxBeatsPerBar = 12

    /// `timeSignature.numerator`, default 4.
    public static func beatsPerBar(_ analysis: Analysis) -> Int {
        min(max(analysis.timeSignature?.numerator ?? 4, 1), maxBeatsPerBar)
    }

    /// Beat indices grouped into bars: a new bar at every downbeat, and after `perBar` beats
    /// (so a missed downbeat never makes a row overflow). Beats before the first downbeat form
    /// a pickup bar.
    public static func groups(_ beats: [Analysis.Beat], perBar: Int) -> [[Int]] {
        let perBar = max(perBar, 1)
        var groups: [[Int]] = []
        var current: [Int] = []
        for index in beats.indices {
            if current.isEmpty == false, beats[index].downbeat || current.count >= perBar {
                groups.append(current)
                current = []
            }
            current.append(index)
        }
        if current.isEmpty == false {
            groups.append(current)
        }
        return groups
    }

    /// Beat index → chord index for each real chord, at the beat nearest its start. When two
    /// chords land on one beat, the later one wins (it's the one that keeps sounding).
    public static func chordStarts(_ chords: [Analysis.Chord], beats: [Analysis.Beat]) -> [Int: Int] {
        guard beats.isEmpty == false else { return [:] }
        var starts: [Int: Int] = [:]
        for (index, chord) in chords.enumerated() where isNoChord(chord.label) == false {
            starts[nearestBeat(to: chord.start, in: beats)] = index
        }
        return starts
    }

    /// Index of the beat nearest `t` (ties go to the earlier beat). `beats` must not be empty.
    public static func nearestBeat(to t: Double, in beats: [Analysis.Beat]) -> Int {
        let after = upperBound(beats, t, \.t)
        if after == 0 { return 0 }
        if after == beats.count { return beats.count - 1 }
        return beats[after].t - t < t - beats[after - 1].t ? after : after - 1
    }

    /// The whole grid. Empty when the analysis has no beats ("No beat grid for this song").
    public static func bars(for analysis: Analysis) -> [BeatGridBar] {
        let beats = analysis.beats
        let chords = analysis.chords
        let starts = chordStarts(chords, beats: beats)
        let grouped = Self.groups(beats, perBar: beatsPerBar(analysis))
        return grouped.enumerated().map { number, group in
            let first = group[0]
            let next = (group.last ?? first) + 1
            let start = beats[first].t
            let end = next < beats.count ? beats[next].t : max(analysis.meta.durationSec, start)
            let cells = group.map { beatIndex in
                let time = beats[beatIndex].t
                let started = starts[beatIndex]
                let sounding = chordIndex(at: time, in: chords).map { isNoChord(chords[$0].label) == false } ?? false
                return BeatGridCell(beatIndex: beatIndex, time: time, chordIndex: started, continues: started == nil && sounding)
            }
            return BeatGridBar(number: number + 1, start: start, end: end, cells: cells)
        }
    }

    /// The chord with `start ≤ t < end`, or `nil`.
    static func chordIndex(at t: Double, in chords: [Analysis.Chord]) -> Int? {
        let index = upperBound(chords, t, \.start) - 1
        return index >= 0 && t < chords[index].end ? index : nil
    }

    /// The bar (index into `bars`) containing beat `beatIndex`, or `nil`.
    public static func barIndex(ofBeat beatIndex: Int, in bars: [BeatGridBar]) -> Int? {
        let index = upperBound(bars, Double(beatIndex)) { Double($0.cells.first?.beatIndex ?? 0) } - 1
        guard bars.indices.contains(index), bars[index].cells.contains(where: { $0.beatIndex == beatIndex }) else {
            return nil
        }
        return index
    }
}
