import Testing
@testable import MusicDecodedCore

/// Fixture landmarks used below:
///     chords[0] N 0–0.72 · [1] A:maj 0.72–1.138 · [2] D:maj 1.138–2.345 · [3] D:maj7 2.345–5.596
///     … [92] C:7 256.209–259.738 · [93] N 259.738–263.407
///     beats 0.337 (downbeat), 0.789, 1.242, 1.672, 2.101 (downbeat), 2.531, … (581 total)
struct TimelineTests {
    let fixture: Analysis
    let synthetic = SyntheticAnalysis.make()

    init() throws {
        fixture = try Contracts.sampleAnalysis()
    }

    // MARK: currentState

    @Test func beforeEverything() {
        #expect(currentState(at: -0.5, in: fixture) == PlaybackState(chordIndex: nil, beatIndex: nil, bar: nil))
    }

    @Test func insideLeadingNoChordBeforeFirstBeat() {
        #expect(currentState(at: 0.1, in: fixture) == PlaybackState(chordIndex: 0, beatIndex: nil, bar: nil))
    }

    @Test func exactlyOnAChordBoundaryBelongsToTheNewChord() {
        #expect(currentState(at: 0.72, in: fixture).chordIndex == 1)
        #expect(currentState(at: 1.138, in: fixture).chordIndex == 2)
        #expect(currentState(at: 0, in: fixture).chordIndex == 0)
    }

    @Test func exactlyOnADownbeatStartsTheNextBar() {
        let state = currentState(at: 2.101, in: fixture)
        #expect(state == PlaybackState(chordIndex: 2, beatIndex: 4, bar: 1))
        #expect(currentState(at: 2.1, in: fixture).bar == 0)
        #expect(currentState(at: 0.337, in: fixture).beatIndex == 0)
    }

    @Test func lastChordAndAfterTheEnd() {
        let lastBeat = fixture.beats.count - 1
        #expect(currentState(at: 263, in: fixture).chordIndex == 93)
        #expect(currentState(at: 263.407, in: fixture).chordIndex == nil, "end is exclusive")
        #expect(currentState(at: 300, in: fixture).chordIndex == nil)
        #expect(currentState(at: 300, in: fixture).beatIndex == lastBeat)
    }

    @Test func barCountsDownbeats() throws {
        let downbeats = fixture.beats.filter(\.downbeat)
        let last = try #require(downbeats.last)
        #expect(currentState(at: last.t, in: fixture).bar == downbeats.count - 1)
    }

    // MARK: upNext

    @Test func upNextSkipsLeadingNoChord() {
        #expect(upNext(at: -1, in: fixture) == UpNext(chordIndex: 1, inBeats: 1))
        #expect(upNext(at: 0.1, in: fixture) == UpNext(chordIndex: 1, inBeats: 1))
    }

    @Test func upNextCountsBeatsUntilTheChord() {
        // A:maj → D:maj at 1.138; no beat in (0.8, 1.138].
        #expect(upNext(at: 0.8, in: fixture) == UpNext(chordIndex: 2, inBeats: 0))
        // D:maj → D:maj7 at 2.345; beats 1.242, 1.672, 2.101.
        #expect(upNext(at: 1.2, in: fixture) == UpNext(chordIndex: 3, inBeats: 3))
    }

    @Test func upNextSkipsRunsOfNoChord() {
        // A:maj, then N, N, then D:maj at 5. Beats 1, 2, 3, 4, 5 are in (0.5, 5].
        #expect(upNext(at: 0.5, in: synthetic) == UpNext(chordIndex: 3, inBeats: 5))
        #expect(upNext(at: 3, in: synthetic) == UpNext(chordIndex: 3, inBeats: 2))
    }

    @Test func upNextIsNilWhenOnlyNoChordRemains() {
        #expect(upNext(at: 257, in: fixture) == nil)
        #expect(upNext(at: 6, in: synthetic) == nil)
        #expect(upNext(at: 500, in: fixture) == nil)
    }

    // MARK: previous / next chord

    @Test func previousRestartsCurrentChordAfterOneSecond() {
        #expect(previousChordStart(at: 3.5, in: fixture) == 2.345)
        #expect(previousChordStart(at: 6.5, in: synthetic) == 5)
    }

    @Test func previousGoesBackWithinTheFirstSecond() {
        #expect(previousChordStart(at: 1.5, in: fixture) == 0.72)
        // Exactly 1 s in is not "more than 1 s", so it goes to the previous chord.
        #expect(previousChordStart(at: 6, in: synthetic) == 0)
    }

    @Test func previousSkipsNoChord() {
        #expect(previousChordStart(at: 5.5, in: synthetic) == 0, "skips N 4–5 and N 2–4")
        #expect(previousChordStart(at: 3, in: synthetic) == 0, "inside N, go to the chord before it")
        #expect(previousChordStart(at: 9.5, in: synthetic) == 5)
        #expect(previousChordStart(at: 261, in: fixture) == 256.209)
    }

    @Test func previousIsNilAtTheStart() {
        #expect(previousChordStart(at: -1, in: fixture) == nil)
        #expect(previousChordStart(at: 0.1, in: fixture) == nil, "only N before")
        #expect(previousChordStart(at: 0.9, in: fixture) == nil, "first chord, under 1 s in")
        #expect(previousChordStart(at: 0.5, in: synthetic) == nil)
    }

    @Test func nextSkipsNoChord() {
        #expect(nextChordStart(at: -1, in: fixture) == 0.72)
        #expect(nextChordStart(at: 0.1, in: fixture) == 0.72)
        #expect(nextChordStart(at: 0.72, in: fixture) == 1.138)
        #expect(nextChordStart(at: 1, in: synthetic) == 5)
    }

    @Test func nextIsNilAfterTheLastChord() {
        #expect(nextChordStart(at: 257, in: fixture) == nil)
        #expect(nextChordStart(at: 5, in: synthetic) == nil)
    }

    // MARK: barRange

    @Test func barRangeIsDownbeatToDownbeat() {
        #expect(barRange(at: 1, in: fixture) == 0.337..<2.101)
        #expect(barRange(at: 2.101, in: fixture) == 2.101..<3.855)
        #expect(barRange(at: 5, in: synthetic) == 4.0..<8.0)
    }

    @Test func barRangeBeforeTheFirstDownbeatIsThePickup() {
        #expect(barRange(at: 0.1, in: fixture) == 0.0..<0.337)
    }

    @Test func barRangeAfterTheLastDownbeatEndsAtTheDuration() throws {
        let last = try #require(fixture.beats.last(where: \.downbeat))
        #expect(barRange(at: 263, in: fixture) == last.t..<263.407)
        #expect(barRange(at: 9, in: synthetic) == 8.0..<10.0)
    }

    @Test func barRangeIsNilWithoutDownbeats() {
        var analysis = synthetic
        analysis.beats = analysis.beats.map { Analysis.Beat(t: $0.t, downbeat: false) }
        #expect(barRange(at: 1, in: analysis) == nil)
    }
}
