import Foundation

/// The clock for a youtube track with no local audio (DESIGN.md §4): the embedded video's
/// `currentTime`. Scrubbing, chord skips and loops seek the video; speed uses `setPlaybackRate`.
///
/// If the video can't play (offline, embedding blocked), it falls back to a silent clock from
/// the last known position, so the chord timeline still works (the offline path, AGENTS.md §5).
@MainActor
final class YouTubePlaybackEngine: PlaybackEngine {
    let video: any VideoPlayerControl
    private let durationSec: Double
    private let now: () -> Double
    private var fallback: SilentPlaybackEngine?

    private(set) var speed = 1.0

    init(
        video: any VideoPlayerControl,
        durationSec: Double,
        now: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }
    ) {
        self.video = video
        self.durationSec = durationSec
        self.now = now
    }

    var videoPlayer: (any VideoPlayerControl)? { video }

    var position: Double {
        if let fallback = activeFallback() {
            return fallback.position
        }
        let t = video.currentTime
        // Like AVPlaybackEngine: once ended, read as the analysis length so end/loop logic fires.
        return video.state == .ended ? max(durationSec, t) : t
    }

    var isPlaying: Bool {
        activeFallback()?.isPlaying ?? video.isPlaying
    }

    func play() {
        if let fallback = activeFallback() {
            fallback.play()
        } else {
            video.play()
        }
    }

    func pause() {
        video.pause()
        fallback?.pause()
    }

    func seek(to seconds: Double) {
        video.seek(to: seconds)
        fallback?.seek(to: seconds)
    }

    func setSpeed(_ speed: Double) {
        self.speed = speed
        video.setPlaybackRate(speed)
        fallback?.setSpeed(speed)
    }

    /// The silent clock once the video has failed, made on first use from where the video was.
    private func activeFallback() -> SilentPlaybackEngine? {
        guard video.hasFailed else { return nil }
        if let fallback {
            return fallback
        }
        let engine = SilentPlaybackEngine(durationSec: durationSec, now: now)
        engine.seek(to: min(video.currentTime, durationSec))
        engine.setSpeed(speed)
        if video.isPlaying {
            engine.play()
        }
        fallback = engine
        return engine
    }
}
