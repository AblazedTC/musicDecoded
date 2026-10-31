import Testing
@testable import MusicDecoded

/// Ports apps/android/.../player/PositionSmootherTest.kt.
struct PositionSmootherTests {
    private let frame = 1.0 / 60

    private func close(_ a: Double, _ b: Double) -> Bool {
        abs(a - b) < 1e-9
    }

    @Test func extrapolatesBetweenCoarseSteps() {
        var smoother = PositionSmoother()
        #expect(close(smoother.position(raw: 10, now: 0, speed: 1), 10))
        #expect(close(smoother.position(raw: 10, now: frame, speed: 1), 10 + frame))
        #expect(close(smoother.position(raw: 10, now: 0.1, speed: 1), 10.1))
    }

    @Test func followsTheSpeed() {
        var smoother = PositionSmoother()
        _ = smoother.position(raw: 10, now: 0, speed: 0.5)
        #expect(close(smoother.position(raw: 10, now: 0.1, speed: 0.5), 10.05))
    }

    @Test func capsHowFarAheadItGuesses() {
        var smoother = PositionSmoother(maxLead: 0.5)
        _ = smoother.position(raw: 10, now: 0, speed: 1)
        #expect(close(smoother.position(raw: 10, now: 3, speed: 1), 10.5))
    }

    @Test func neverStepsBackALittle() {
        var smoother = PositionSmoother()
        _ = smoother.position(raw: 10, now: 0, speed: 1)
        #expect(close(smoother.position(raw: 10, now: 0.25, speed: 1), 10.25))
        // The next step lands just behind the estimate: hold, then move on from the new step.
        #expect(close(smoother.position(raw: 10.2, now: 0.26, speed: 1), 10.25))
        #expect(close(smoother.position(raw: 10.2, now: 0.36, speed: 1), 10.3))
    }

    @Test func bigJumpsPassThrough() {
        var smoother = PositionSmoother()
        _ = smoother.position(raw: 10, now: 0, speed: 1)
        _ = smoother.position(raw: 10, now: 0.2, speed: 1)
        #expect(close(smoother.position(raw: 3, now: 0.21, speed: 1), 3))
        #expect(close(smoother.position(raw: 42, now: 0.22, speed: 1), 42))
    }

    @Test func resetForgetsTheLastValue() {
        var smoother = PositionSmoother()
        _ = smoother.position(raw: 10, now: 0, speed: 1)
        _ = smoother.position(raw: 10, now: 0.2, speed: 1)
        smoother.reset()
        #expect(close(smoother.position(raw: 10.1, now: 0.21, speed: 1), 10.1))
    }
}
