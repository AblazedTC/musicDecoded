import Foundation

/// The embedded YouTube player as the playback engines see it. `YouTubePlayerController` is the
/// real one; tests use a fake.
@MainActor
protocol VideoPlayerControl: AnyObject {
    /// The IFrame player accepted commands. Commands sent before are applied once it's ready.
    var isReady: Bool { get }
    /// The player can't play (offline, embedding disabled, removed video).
    var hasFailed: Bool { get }
    var state: YouTubePlayerState { get }
    /// `true` from `play()` (or a playing/buffering report) until `pause()` (or a paused report).
    /// Unlike `state`, it flips right away, so the clock doesn't stop while the command travels.
    var isPlaying: Bool { get }
    /// Seconds into the video, extrapolated between reports.
    var currentTime: Double { get }

    func play()
    func pause()
    func seek(to seconds: Double)
    func setPlaybackRate(_ rate: Double)
    func setMuted(_ muted: Bool)
}
