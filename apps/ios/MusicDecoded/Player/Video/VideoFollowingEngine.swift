import Foundation

/// A youtube track with local audio (only when `offlineStreamAudio` allowed saving it, PLAN.md R-1):
/// the audio is the clock (DESIGN.md §4) and the video follows it, muted, within 150 ms.
/// `synchronize()` runs on every clock tick and re-seeks the video when it drifts further.
@MainActor
final class VideoFollowingEngine: PlaybackEngine {
    /// The most the video may drift from the audio (T-305 acceptance).
    static let maxDrift = 0.15
    /// A re-seek needs time to land; don't seek again sooner than this.
    static let correctionCooldown = 1.0

    let audio: any PlaybackEngine
    let video: any VideoPlayerControl
    private let now: () -> Double
    private var lastCorrection = -Double.infinity

    init(
        audio: any PlaybackEngine,
        video: any VideoPlayerControl,
        now: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }
    ) {
        self.audio = audio
        self.video = video
        self.now = now
        video.setMuted(true)
        video.setPlaybackRate(audio.speed)
    }

    var videoPlayer: (any VideoPlayerControl)? { video }

    var position: Double { audio.position }
    var isPlaying: Bool { audio.isPlaying }
    var speed: Double { audio.speed }

    func play() {
        audio.play()
        video.play()
    }

    func pause() {
        audio.pause()
        video.pause()
    }

    func seek(to seconds: Double) {
        audio.seek(to: seconds)
        video.seek(to: seconds)
        lastCorrection = now()
    }

    func setSpeed(_ speed: Double) {
        audio.setSpeed(speed)
        video.setPlaybackRate(speed)
    }

    /// Keeps the video's play state and position on the audio's.
    func synchronize() {
        guard video.isReady, video.hasFailed == false else { return }
        let playing = audio.isPlaying
        if playing != video.isPlaying {
            if playing {
                video.play()
            } else {
                video.pause()
            }
        }
        // While buffering the video's time stands still; measuring drift then would re-seek forever.
        guard video.state != .buffering else { return }
        let target = audio.position
        let t = now()
        if abs(video.currentTime - target) > Self.maxDrift, t - lastCorrection >= Self.correctionCooldown {
            video.seek(to: target)
            lastCorrection = t
        }
    }
}
