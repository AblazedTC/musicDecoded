import AVFoundation
import Foundation

/// What the player's clock reads from. All times are seconds.
@MainActor
protocol PlaybackEngine: AnyObject {
    /// Current playback position.
    var position: Double { get }
    /// `true` while playing (or about to, e.g. buffering). The engine may stop on its own, e.g.
    /// when another app interrupts the audio session.
    var isPlaying: Bool { get }
    /// The last `setSpeed` value (1 until set), so a reopened player picks it up.
    var speed: Double { get }

    func play()
    func pause()
    func seek(to seconds: Double)
    /// 0.5–1.25×, pitch kept.
    func setSpeed(_ speed: Double)

    /// Runs on every clock tick. Engines that keep a second player in step (the muted video
    /// following local audio) correct it here.
    func synchronize()
    /// The embedded YouTube player for the Video panel, or `nil` when the engine has none.
    var videoPlayer: (any VideoPlayerControl)? { get }
}

extension PlaybackEngine {
    func synchronize() {}

    var videoPlayer: (any VideoPlayerControl)? { nil }
}

/// Plays the local `audio/{trackId}` file with `AVPlayer`, time-stretched without changing pitch
/// (`audioTimePitchAlgorithm = .timeDomain`, DESIGN.md §4). Once the item has ended the position
/// reads as `durationSec` (the analysis length), so the clock's end/loop logic always fires.
@MainActor
final class AVPlaybackEngine: PlaybackEngine {
    private let player: AVPlayer
    private let durationSec: Double
    /// The target of a seek still in flight: `currentTime()` can lag behind until it completes.
    private var pendingSeek: Double?
    private var seekGeneration = 0

    private(set) var speed = 1.0

    init(url: URL, durationSec: Double) {
        let item = AVPlayerItem(url: url)
        item.audioTimePitchAlgorithm = .timeDomain
        player = AVPlayer(playerItem: item)
        // Stay "playing" at the end so a loop that ends with the track can wrap around.
        player.actionAtItemEnd = .none
        self.durationSec = durationSec
    }

    var position: Double {
        if let pendingSeek { return pendingSeek }
        let current = player.currentTime().seconds
        let t = current.isFinite ? current : 0
        if let item = player.currentItem, item.duration.isNumeric, t >= item.duration.seconds - Self.endTolerance {
            return max(durationSec, t)
        }
        return t
    }

    var isPlaying: Bool {
        player.rate != 0
    }

    func play() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default)
        try? session.setActive(true)
        player.play()
    }

    func pause() {
        player.pause()
    }

    func seek(to seconds: Double) {
        seekGeneration += 1
        let generation = seekGeneration
        pendingSeek = seconds
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
            let engine = self
            Task { @MainActor in
                engine?.seekFinished(generation)
            }
        }
    }

    func setSpeed(_ speed: Double) {
        self.speed = speed
        // `defaultRate` is the rate `play()` uses; a playing player switches right away.
        player.defaultRate = Float(speed)
        if player.rate != 0 {
            player.rate = Float(speed)
        }
    }

    private func seekFinished(_ generation: Int) {
        if generation == seekGeneration {
            pendingSeek = nil
        }
    }

    /// AVPlayer can stop a few milliseconds short of the item's duration.
    private static let endTolerance = 0.05
}

/// A clock with no audio, for file/recording tracks whose audio isn't on this device, the debug
/// fixture, and the fallback when a YouTube video can't play. Advances in real time × speed; `now` (seconds,
/// monotonic) is injectable so tests can drive it.
@MainActor
final class SilentPlaybackEngine: PlaybackEngine {
    private let durationSec: Double
    private let now: () -> Double
    private var anchorPosition = 0.0
    private var anchorTime = 0.0
    private(set) var speed = 1.0
    private(set) var isPlaying = false

    init(durationSec: Double, now: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }) {
        self.durationSec = durationSec
        self.now = now
    }

    var position: Double {
        guard isPlaying else { return anchorPosition }
        return min(anchorPosition + (now() - anchorTime) * speed, durationSec)
    }

    func play() {
        anchorTime = now()
        isPlaying = true
    }

    func pause() {
        anchorPosition = position
        isPlaying = false
    }

    func seek(to seconds: Double) {
        anchorPosition = min(max(seconds, 0), durationSec)
        anchorTime = now()
    }

    func setSpeed(_ speed: Double) {
        anchorPosition = position
        anchorTime = now()
        self.speed = speed
    }
}
