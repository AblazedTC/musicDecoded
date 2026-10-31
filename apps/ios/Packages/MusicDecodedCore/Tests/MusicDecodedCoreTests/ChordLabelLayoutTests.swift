import Testing
@testable import MusicDecodedCore

/// DESIGN.md §3.7 item 3: two staggered rows, 4pt gap, drop on collision, current never dropped.
struct ChordLabelLayoutTests {
    private func box(_ id: Int, _ x: Double, _ width: Double = 40) -> ChordLabelBox {
        ChordLabelBox(id: id, x: x, width: width)
    }

    @Test func labelsThatFitStayInTheFirstRow() {
        let rows = ChordLabelLayout.layout([box(0, 0), box(1, 44), box(2, 100)], current: nil)
        #expect(rows == [0: 0, 1: 0, 2: 0])
    }

    @Test func theGapIsFourPoints() {
        #expect(ChordLabelLayout.layout([box(0, 0), box(1, 44)], current: nil)[1] == 0)
        #expect(ChordLabelLayout.layout([box(0, 0), box(1, 43.9)], current: nil)[1] == 1)
    }

    @Test func anOverlappingLabelMovesToTheSecondRow() {
        let rows = ChordLabelLayout.layout([box(0, 0), box(1, 20), box(2, 50)], current: nil)
        #expect(rows == [0: 0, 1: 1, 2: 0])
    }

    @Test func aLabelThatFitsNoRowIsDropped() {
        let rows = ChordLabelLayout.layout([box(0, 0), box(1, 10), box(2, 20)], current: nil)
        #expect(rows == [0: 0, 1: 1])
    }

    @Test func theCurrentChordReplacesItsNeighbor() {
        let rows = ChordLabelLayout.layout([box(0, 0), box(1, 10), box(2, 20)], current: 2)
        // Row 0's label (ends at 40) ends before row 1's (ends at 50), so it gives way.
        #expect(rows == [1: 1, 2: 0])
        #expect(rows[2] != nil)
    }

    @Test func labelsAfterTheCurrentOneYieldToIt() {
        let rows = ChordLabelLayout.layout([box(0, 0), box(1, 10), box(2, 20), box(3, 30)], current: 2)
        #expect(rows[2] != nil)
        #expect(rows[3] == nil)
    }

    @Test func labelsThatFitNeedOneRow() {
        #expect(ChordLabelLayout.rowCount([box(0, 0), box(1, 44), box(2, 100)]) == 1)
    }

    @Test func anOverlapAnywhereNeedsTheSecondRow() {
        #expect(ChordLabelLayout.rowCount([box(0, 0), box(1, 44), box(2, 60), box(3, 500)]) == 2)
    }

    @Test func noLabelsStillHaveOneRow() {
        #expect(ChordLabelLayout.rowCount([]) == 1)
    }

    @Test func theRowCountFollowsTheZoom() {
        let chords = [
            Analysis.Chord(start: 0, end: 0.2, label: "A:maj"),
            Analysis.Chord(start: 0.2, end: 0.4, label: "D:maj"),
        ]
        let narrow = ChordLabelLayout.boxes(for: chords, pointsPerSecond: 60) { _ in 40 }
        #expect(ChordLabelLayout.rowCount(narrow) == 2)
        let wide = ChordLabelLayout.boxes(for: chords, pointsPerSecond: 400) { _ in 40 }
        #expect(ChordLabelLayout.rowCount(wide) == 1)
    }

    /// The band mustn't jump during playback: when one row is enough, no current chord uses another.
    @Test func noCurrentChordUsesARowTheBandDoesntHave() {
        let boxes = [box(0, 0), box(1, 44), box(2, 100)]
        #expect(ChordLabelLayout.rowCount(boxes) == 1)
        for current in boxes.map(\.id) {
            #expect(ChordLabelLayout.layout(boxes, current: current).values.allSatisfy { $0 == 0 })
        }
    }

    @Test func zoomingInBringsDroppedLabelsBack() {
        let chords = [
            Analysis.Chord(start: 0, end: 0.2, label: "A:maj"),
            Analysis.Chord(start: 0.2, end: 0.4, label: "N"),
            Analysis.Chord(start: 0.4, end: 0.6, label: "D:maj"),
            Analysis.Chord(start: 0.6, end: 1, label: "E:maj"),
        ]
        let narrow = ChordLabelLayout.boxes(for: chords, pointsPerSecond: 60) { _ in 40 }
        #expect(narrow.map(\.id) == [0, 2, 3])
        #expect(ChordLabelLayout.layout(narrow, current: nil).count == 2)

        let wide = ChordLabelLayout.boxes(for: chords, pointsPerSecond: 400) { _ in 40 }
        #expect(ChordLabelLayout.layout(wide, current: nil) == [0: 0, 2: 0, 3: 0])
    }
}
