/// A chord label box on the timeline, in content points (x = start × points per second).
public struct ChordLabelBox: Sendable, Equatable {
    /// The chord's index in `analysis.chords`.
    public let id: Int
    /// The left edge, anchored at the chord change.
    public let x: Double
    public let width: Double

    public init(id: Int, x: Double, width: Double) {
        self.id = id
        self.x = x
        self.width = width
    }

    public var right: Double { x + width }
}

/// Timeline chord-label collision handling (DESIGN.md §3.7 item 3): labels go in time order into
/// up to `rows` staggered rows, each in the first row where it clears the previous label of that
/// row by `gap`. A label that fits no row is dropped at this zoom, except the current chord's,
/// which replaces the overlapping neighbor. Re-run whenever the zoom (or current chord) changes.
public enum ChordLabelLayout {
    public static let rows = 2
    public static let gap: Double = 4

    /// Boxes for every chord except `N`/`X`, at `pointsPerSecond`; `width` measures chord `i`'s label.
    public static func boxes(
        for chords: [Analysis.Chord],
        pointsPerSecond: Double,
        width: (Int) -> Double
    ) -> [ChordLabelBox] {
        chords.indices.compactMap { index in
            let chord = chords[index]
            guard isNoChord(chord.label) == false else { return nil }
            return ChordLabelBox(id: index, x: chord.start * pointsPerSecond, width: width(index))
        }
    }

    /// How many rows the label band needs at this zoom, at least 1. Depends only on the boxes (the
    /// analysis and zoom), never on the playhead: a current chord only ever takes the place of a
    /// label in a row that's already in use, so the band can't change during playback.
    public static func rowCount(_ boxes: [ChordLabelBox], rows: Int = rows, gap: Double = gap) -> Int {
        (layout(boxes, current: nil, rows: rows, gap: gap).values.max() ?? 0) + 1
    }

    /// Row (0-based) per placed box id; dropped ids are missing.
    public static func layout(
        _ boxes: [ChordLabelBox],
        current: Int?,
        rows: Int = rows,
        gap: Double = gap
    ) -> [Int: Int] {
        guard rows > 0 else { return [:] }
        let ordered = boxes.enumerated().sorted { a, b in
            a.element.x == b.element.x ? a.offset < b.offset : a.element.x < b.element.x
        }.map(\.element)
        var lastInRow: [ChordLabelBox?] = Array(repeating: nil, count: rows)
        var placed: [Int: Int] = [:]

        for box in ordered {
            if let row = lastInRow.firstIndex(where: { last in last.map { box.x >= $0.right + gap } ?? true }) {
                lastInRow[row] = box
                placed[box.id] = row
            } else if box.id == current {
                // Never drop the current chord: it replaces the neighbor that ends first.
                var row = 0
                for candidate in 1..<rows {
                    if let a = lastInRow[candidate], let b = lastInRow[row], a.right < b.right {
                        row = candidate
                    }
                }
                if let neighbor = lastInRow[row] {
                    placed[neighbor.id] = nil
                }
                lastInRow[row] = box
                placed[box.id] = row
            }
        }
        return placed
    }
}
