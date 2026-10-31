@testable import MusicDecoded

/// A `VideoPlayerControl` that records commands; tests set what it reports.
@MainActor
final class FakeVideoPlayer: VideoPlayerControl {
    var isReady = true
    var hasFailed = false
    var state = YouTubePlayerState.paused
    var isPlaying = false
    var currentTime = 0.0

    private(set) var seeks: [Double] = []
    private(set) var rates: [Double] = []
    private(set) var muted: Bool?
    private(set) var playCalls = 0
    private(set) var pauseCalls = 0

    func play() {
        playCalls += 1
        isPlaying = true
    }

    func pause() {
        pauseCalls += 1
        isPlaying = false
    }

    func seek(to seconds: Double) {
        seeks.append(seconds)
        currentTime = seconds
    }

    func setPlaybackRate(_ rate: Double) {
        rates.append(rate)
    }

    func setMuted(_ muted: Bool) {
        self.muted = muted
    }
}
