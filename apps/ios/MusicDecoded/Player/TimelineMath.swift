import CoreGraphics
import MusicDecodedCore

// Pure timeline geometry (DESIGN.md §3.7 item 3): the playhead is fixed at the horizontal center
// and content scrolls under it, so x = center + (t − playhead) × pointsPerSecond.

enum TimelineZoom {
    /// Zoom limits in points per second (DESIGN.md §3.7: 60–400, default 120).
    static let minimum: CGFloat = 60
    static let maximum: CGFloat = 400
    static let standard: CGFloat = 120

    static func clamp(_ pointsPerSecond: CGFloat) -> CGFloat {
        min(max(pointsPerSecond, minimum), maximum)
    }
}

/// Maps time ↔ x for a viewport `width` points wide whose center is the playhead at `playhead` seconds.
struct TimelineViewport: Equatable, Sendable {
    let width: CGFloat
    let playhead: Double
    let pointsPerSecond: CGFloat

    var center: CGFloat { width / 2 }
    var startTime: Double { playhead - Double(center / pointsPerSecond) }
    var endTime: Double { playhead + Double(center / pointsPerSecond) }

    func x(_ t: Double) -> CGFloat {
        center + CGFloat(t - playhead) * pointsPerSecond
    }

    func time(_ x: CGFloat) -> Double {
        playhead + Double((x - center) / pointsPerSecond)
    }
}

enum TimelineMath {
    /// Indices of the chords that overlap `from...to`, so only the visible window is drawn.
    static func visibleChordIndices(_ chords: [Analysis.Chord], from: Double, to: Double) -> Range<Int> {
        guard chords.isEmpty == false else { return 0..<0 }
        let first = max(upperBound(chords, from, \.start) - 1, 0)
        let end = upperBound(chords, to, \.start)
        return first < end ? first..<end : first..<first
    }

    /// Indices of the beats with `from ≤ t ≤ to`.
    static func visibleBeatIndices(_ beats: [Analysis.Beat], from: Double, to: Double) -> Range<Int> {
        let first = upperBound(beats, from - 1e-9, \.t)
        let end = upperBound(beats, to, \.t)
        return first < end ? first..<end : first..<first
    }

    /// Indices of the waveform peaks (one per `1/fps` s) inside `from...to`.
    static func visiblePeakIndices(_ peaks: Analysis.WaveformPeaks, from: Double, to: Double) -> Range<Int> {
        let count = peaks.peaks.count
        guard count > 0, peaks.fps > 0, to >= 0, from * peaks.fps < Double(count) else { return 0..<0 }
        let first = min(max(Int(from * peaks.fps), 0), count - 1)
        let last = min(max(Int(to * peaks.fps), 0), count - 1)
        return first <= last ? first..<(last + 1) : first..<first
    }

    /// The beat time nearest to `t`; `t` itself when there are no beats.
    static func snapToBeat(_ t: Double, beats: [Analysis.Beat]) -> Double {
        guard beats.isEmpty == false else { return t }
        let after = upperBound(beats, t, \.t)
        var candidates: [Double] = []
        if after > 0 { candidates.append(beats[after - 1].t) }
        if after < beats.count { candidates.append(beats[after].t) }
        return candidates.min { abs($0 - t) < abs($1 - t) } ?? t
    }

    /// A custom loop dragged between `a` and `b` (either order), snapped to beats (DESIGN.md §4).
    /// A drag that snaps to a single beat extends to the next beat. `nil` if nothing usable is left.
    static func snappedLoop(_ a: Double, _ b: Double, in analysis: Analysis) -> Range<Double>? {
        let duration = analysis.meta.durationSec
        var start = clamp(snapToBeat(min(a, b), beats: analysis.beats), 0, duration)
        var end = clamp(snapToBeat(max(a, b), beats: analysis.beats), 0, duration)
        if end <= start {
            end = analysis.beats.first(where: { $0.t > start })?.t ?? duration
        }
        if end <= start {
            start = max(end - 1, 0)
        }
        return start < end ? start..<end : nil
    }

    /// Seconds between time labels so they are at least `minSpacing` points apart.
    static func timeLabelStep(pointsPerSecond: CGFloat, minSpacing: CGFloat) -> Int {
        labelSteps.first { CGFloat($0) * pointsPerSecond >= minSpacing } ?? labelSteps[labelSteps.count - 1]
    }

    private static let labelSteps = [1, 2, 5, 10, 15, 30, 60]

    private static func clamp(_ value: Double, _ low: Double, _ high: Double) -> Double {
        min(max(value, low), high)
    }
}
