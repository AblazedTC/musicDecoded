import CoreGraphics
import Foundation
import MusicDecodedCore
import Testing
@testable import MusicDecoded

/// Ports apps/android/.../player/TimelineMathTest.kt.
struct TimelineMathTests {
    let synthetic: Analysis

    init() throws {
        synthetic = try PlayerFixtures.synthetic()
    }

    @Test func viewportKeepsThePlayheadAtTheCenter() {
        let viewport = TimelineViewport(width: 400, playhead: 10, pointsPerSecond: 100)
        #expect(abs(viewport.x(10) - 200) < 0.001)
        #expect(abs(viewport.x(11) - 300) < 0.001)
        #expect(abs(viewport.startTime - 8) < 1e-9)
        #expect(abs(viewport.endTime - 12) < 1e-9)
        #expect(abs(viewport.time(350) - 11.5) < 1e-9)
    }

    @Test func zoomIsClampedTo60to400() {
        #expect(TimelineZoom.clamp(10) == 60)
        #expect(TimelineZoom.clamp(1000) == 400)
        #expect(TimelineZoom.clamp(TimelineZoom.standard) == 120)
    }

    @Test func visibleChordsAreOnlyThoseOverlappingTheWindow() {
        // A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10
        #expect(TimelineMath.visibleChordIndices(synthetic.chords, from: 3, to: 6) == 1..<4)
        #expect(TimelineMath.visibleChordIndices(synthetic.chords, from: -5, to: 1) == 0..<1)
        #expect(TimelineMath.visibleChordIndices(synthetic.chords, from: 9, to: 20) == 4..<5)
        #expect(TimelineMath.visibleChordIndices([], from: 0, to: 1).isEmpty)
    }

    @Test func visibleBeatsAreInclusive() {
        #expect(TimelineMath.visibleBeatIndices(synthetic.beats, from: 2, to: 5) == 2..<6)
        #expect(TimelineMath.visibleBeatIndices(synthetic.beats, from: -3, to: -1).isEmpty)
    }

    @Test func visiblePeaksFollowFps() throws {
        let peaks = try waveform(fps: 10, count: 100)
        #expect(TimelineMath.visiblePeakIndices(peaks, from: 2, to: 4) == 20..<41)
        #expect(TimelineMath.visiblePeakIndices(peaks, from: -3, to: 0.5) == 0..<6)
        #expect(TimelineMath.visiblePeakIndices(peaks, from: 9, to: 30) == 90..<100)
        #expect(TimelineMath.visiblePeakIndices(peaks, from: -3, to: -1).isEmpty)
        #expect(TimelineMath.visiblePeakIndices(peaks, from: 11, to: 12).isEmpty)
    }

    @Test func sunnyDaysFixtureDrawsAFewDozenPeaksPerScreen() throws {
        let analysis = try PlayerFixtures.sunnyDays()
        let peaks = try #require(analysis.waveformPeaks)
        // 400 pt wide at 120 pt/s ≈ 3.3 s visible → ~67 bars, not all of them.
        let viewport = TimelineViewport(width: 400, playhead: 60, pointsPerSecond: 120)
        let visible = TimelineMath.visiblePeakIndices(peaks, from: viewport.startTime, to: viewport.endTime)
        #expect((60...70).contains(visible.count))
        #expect(visible.count < peaks.peaks.count)
    }

    @Test func snapsToTheNearestBeat() {
        #expect(TimelineMath.snapToBeat(3.4, beats: synthetic.beats) == 3)
        #expect(TimelineMath.snapToBeat(3.6, beats: synthetic.beats) == 4)
        #expect(TimelineMath.snapToBeat(-2, beats: synthetic.beats) == 0)
        #expect(TimelineMath.snapToBeat(42, beats: synthetic.beats) == 9)
        #expect(TimelineMath.snapToBeat(1.7, beats: []) == 1.7)
    }

    @Test func draggedLoopSnapsInEitherDirection() {
        #expect(TimelineMath.snappedLoop(5.8, 2.2, in: synthetic) == 2..<6)
        #expect(TimelineMath.snappedLoop(2.2, 5.8, in: synthetic) == 2..<6)
    }

    @Test func tinyDragExtendsToTheNextBeat() {
        #expect(TimelineMath.snappedLoop(3.1, 3.2, in: synthetic) == 3..<4)
    }

    @Test func dragPastTheLastBeatEndsAtTheDuration() {
        #expect(TimelineMath.snappedLoop(9.2, 9.3, in: synthetic) == 9..<10)
    }

    @Test func noBeatsStillGivesALoop() throws {
        let noBeats = try PlayerFixtures.synthetic(beats: false)
        #expect(TimelineMath.snappedLoop(2.5, 4, in: noBeats) == 2.5..<4)
        let empty = try PlayerFixtures.synthetic(beats: false, durationSec: 0)
        #expect(TimelineMath.snappedLoop(-5, -1, in: empty) == nil)
    }

    @Test func labelStepKeepsLabelsApart() {
        #expect(TimelineMath.timeLabelStep(pointsPerSecond: 120, minSpacing: 56) == 1)
        #expect(TimelineMath.timeLabelStep(pointsPerSecond: 40, minSpacing: 56) == 2)
        #expect(TimelineMath.timeLabelStep(pointsPerSecond: 0.1, minSpacing: 56) == 60)
    }

    @Test func timelineTakesAThirdOfTheMiddleClamped() {
        #expect(abs(TimelineMath.timelineHeight(available: 500) - 160) < 0.001)
        #expect(TimelineMath.timelineHeight(available: 300) == 150)
        #expect(TimelineMath.timelineHeight(available: 1000) == 220)
    }

    @Test func formatsTimes() {
        #expect(PlayerFormat.time(0) == "0:00")
        #expect(PlayerFormat.time(64.9) == "1:04")
        #expect(PlayerFormat.time(-3) == "0:00")
        #expect(PlayerFormat.loop(64..<80) == "1:04–1:20")
    }

    private func waveform(fps: Double, count: Int) throws -> Analysis.WaveformPeaks {
        let peaks = Array(repeating: "0.5", count: count).joined(separator: ",")
        let json = #"{"fps": \#(fps), "peaks": [\#(peaks)]}"#
        return try JSONDecoder().decode(Analysis.WaveformPeaks.self, from: Data(json.utf8))
    }
}
