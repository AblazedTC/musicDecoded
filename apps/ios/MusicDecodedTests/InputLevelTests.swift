import Testing
@testable import MusicDecoded

/// The recorder meter's math (the recording itself needs a microphone and isn't unit-tested).
struct InputLevelTests {
    @Test func clampsToTheMeterRange() {
        #expect(InputLevel.clamp(-160) == InputLevel.minDb)
        #expect(InputLevel.clamp(3) == 0)
        #expect(InputLevel.clamp(-12) == -12)
        #expect(InputLevel.clamp(-.infinity) == InputLevel.minDb)
        #expect(InputLevel.clamp(.nan) == InputLevel.minDb)
    }

    @Test(arguments: [(Float(-60), 0), (-30, 8), (0, 16), (-160, 0)])
    func litSegments(db: Float, lit: Int) {
        #expect(InputLevel.litSegments(db) == lit)
    }

    @Test func normalizedLevel() {
        #expect(InputLevel.normalized(-60) == 0)
        #expect(InputLevel.normalized(-30) == 0.5)
        #expect(InputLevel.normalized(0) == 1)
    }

    @Test func timerFormat() {
        #expect(InputLevel.formatTimer(0) == "00:00:00")
        #expect(InputLevel.formatTimer(83.9) == "00:01:23")
        #expect(InputLevel.formatTimer(600) == "00:10:00")
        #expect(InputLevel.formatTimer(3_725) == "01:02:05")
    }
}
