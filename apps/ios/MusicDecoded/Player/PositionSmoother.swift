import Foundation

/// Turns a position that only updates in steps into a smooth clock (ports Android's
/// `PositionSmoother`): between steps it extrapolates in real time × speed, at most `maxLead`
/// past the last step (in case playback stalls), and it never steps back by less than
/// `backTolerance`, because the next step often lands just behind the estimate. Bigger jumps
/// (seeks, loop wraps) pass through. Call `reset()` on play, pause, seek and speed changes.
/// All times are seconds; `now` is monotonic.
struct PositionSmoother: Sendable {
    let maxLead: Double
    let backTolerance: Double

    private var stepValue: Double?
    private var stepTime = 0.0
    private var last: Double?

    init(maxLead: Double = 0.5, backTolerance: Double = 0.25) {
        self.maxLead = maxLead
        self.backTolerance = backTolerance
    }

    mutating func reset() {
        stepValue = nil
        last = nil
    }

    mutating func position(raw: Double, now: Double, speed: Double) -> Double {
        if raw != stepValue {
            stepValue = raw
            stepTime = now
        }
        var t = raw + min(max((now - stepTime) * speed, 0), maxLead)
        if let last, t < last, last - t < backTolerance {
            t = last
        }
        last = t
        return t
    }
}
