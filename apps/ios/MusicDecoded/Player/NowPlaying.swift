import Foundation
import MusicDecodedCore
import Observation

/// The app-level "now playing" session: owns the `PlaybackEngine` of the last track opened in the
/// Player, so playback keeps going when the Player is closed and the mini player (DESIGN.md §2) can
/// show and control it from Home and Library. Ports apps/android/.../player/NowPlaying.kt.
@MainActor
@Observable
final class NowPlaying {
    /// What the mini player shows.
    struct Track: Equatable, Sendable {
        let trackId: String
        let title: String
        let artist: String?
        let key: Analysis.Key?
        let durationSec: Double
    }

    /// `nil` when nothing is loaded (no mini player).
    private(set) var track: Track?
    private(set) var isPlaying = false
    /// Seconds; updated by `tick()`.
    private(set) var position = 0.0

    @ObservationIgnored private var engine: (any PlaybackEngine)?

    private static let endTolerance = 0.05

    /// The engine for `trackId`: the running one if that track is already loaded; otherwise the
    /// previous one is stopped and `create` makes a new one.
    func engine(
        for trackId: String,
        track playerTrack: PlayerTrack,
        create: (PlayerTrack) -> any PlaybackEngine
    ) -> any PlaybackEngine {
        if let engine, track?.trackId == trackId {
            return engine
        }
        engine?.pause()
        let created = create(playerTrack)
        engine = created
        track = Track(
            trackId: trackId,
            title: playerTrack.title,
            artist: playerTrack.artist,
            key: playerTrack.analysis.key,
            durationSec: playerTrack.analysis.meta.durationSec
        )
        position = created.position
        isPlaying = created.isPlaying
        return created
    }

    /// Reads the engine clock; stops at the end of the track (loops only run inside the Player).
    func tick() {
        guard let engine, let track else { return }
        engine.synchronize()
        var t = engine.position
        if engine.isPlaying, t >= track.durationSec {
            engine.pause()
            t = track.durationSec
        }
        if t != position {
            position = t
        }
        if engine.isPlaying != isPlaying {
            isPlaying = engine.isPlaying
        }
    }

    func togglePlay() {
        guard let engine, let track else { return }
        if engine.isPlaying {
            engine.pause()
        } else {
            if engine.position >= track.durationSec - Self.endTolerance {
                engine.seek(to: 0)
            }
            engine.play()
        }
        tick()
    }

    func pause() {
        if engine?.isPlaying == true {
            togglePlay()
        }
    }

    /// The mini player's ×: stops and unloads the track.
    func dismiss() {
        engine?.pause()
        engine = nil
        track = nil
        isPlaying = false
        position = 0
    }

    /// Unloads `trackId` if it's the one loaded (e.g. it was just deleted).
    func stop(trackId: String) {
        if track?.trackId == trackId {
            dismiss()
        }
    }
}
