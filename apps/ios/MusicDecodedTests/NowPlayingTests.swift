import Foundation
import MusicDecodedCore
import Testing
@testable import MusicDecoded

/// The app-level session behind the mini player, with a fake-clock `SilentPlaybackEngine`.
/// Ports apps/android/.../player/NowPlayingTest.kt.
@MainActor
struct NowPlayingTests {
    final class Clock {
        var now = 0.0
    }

    let clock = Clock()
    let nowPlaying = NowPlaying()
    let track: PlayerTrack

    init() throws {
        // 10 s synthetic analysis.
        track = PlayerTrack(analysis: try PlayerFixtures.synthetic(), title: "Song", artist: "Artist", favorite: false, audioURL: nil)
    }

    private static func engine(_ track: PlayerTrack, clock: Clock) -> any PlaybackEngine {
        SilentPlaybackEngine(durationSec: track.analysis.meta.durationSec, now: { clock.now })
    }

    private func makeEngine(_ track: PlayerTrack) -> any PlaybackEngine {
        Self.engine(track, clock: clock)
    }

    /// A player model wired the way `PlayerScreen` does it (engine from `NowPlaying`).
    private func playerModel(trackId: String = "t1") async -> PlayerModel {
        let track = track
        let nowPlaying = nowPlaying
        let clock = clock
        let model = PlayerModel(
            loadTrack: { track },
            voicings: nil,
            makeEngine: { loaded in
                nowPlaying.engine(for: trackId, track: loaded) { Self.engine($0, clock: clock) }
            },
            tickInterval: nil
        )
        await model.load()
        return model
    }

    @Test func playbackOutlivesThePlayerAndReopeningReusesTheEngine() async {
        var first: PlayerModel? = await playerModel()
        first?.togglePlay()
        first?.setSpeed(0.75)
        clock.now += 2
        #expect(nowPlaying.track?.title == "Song")

        // Leave the player: the engine keeps going and the mini player keeps the clock.
        first = nil
        nowPlaying.tick()
        #expect(nowPlaying.isPlaying)
        #expect(abs(nowPlaying.position - 1.5) < 1e-6)

        // Reopen: same engine; playing, speed and position carry over.
        let second = await playerModel()
        #expect(second.isPlaying)
        #expect(second.speed == 0.75)
        #expect(abs(second.position - 1.5) < 1e-6)
    }

    @Test func anotherTrackReplacesAndStopsTheOldEngine() {
        let engineA = nowPlaying.engine(for: "a", track: track, create: makeEngine)
        #expect(nowPlaying.engine(for: "a", track: track, create: makeEngine) === engineA)
        engineA.play()

        let engineB = nowPlaying.engine(for: "b", track: track, create: makeEngine)
        #expect(engineB !== engineA)
        #expect(engineA.isPlaying == false)
        #expect(nowPlaying.track?.trackId == "b")
    }

    @Test func miniPlayerControlsAndEnd() {
        _ = nowPlaying.engine(for: "a", track: track, create: makeEngine)
        nowPlaying.togglePlay()
        #expect(nowPlaying.isPlaying)
        clock.now += 100 // past the 10 s track
        nowPlaying.tick()
        #expect(nowPlaying.isPlaying == false)
        #expect(abs(nowPlaying.position - 10) < 1e-6)

        nowPlaying.togglePlay() // at the end: restarts from the top
        #expect(nowPlaying.isPlaying)
        #expect(nowPlaying.position == 0)

        nowPlaying.pause()
        #expect(nowPlaying.isPlaying == false)
        nowPlaying.dismiss()
        #expect(nowPlaying.track == nil)
    }

    @Test func deletingTheLoadedTrackStopsAndClearsIt() {
        let engine = nowPlaying.engine(for: "a", track: track, create: makeEngine)
        nowPlaying.togglePlay()

        nowPlaying.stop(trackId: "other")
        #expect(nowPlaying.track?.trackId == "a")

        nowPlaying.stop(trackId: "a")
        #expect(nowPlaying.track == nil)
        #expect(nowPlaying.isPlaying == false)
        #expect(engine.isPlaying == false)
    }

    @Test func subtitleIsArtistAndKey() {
        let withKey = NowPlaying.Track(trackId: "a", title: "Song", artist: "Artist", key: nil, durationSec: 10)
        #expect(MiniPlayer.subtitle(withKey) == "Artist")
        let none = NowPlaying.Track(trackId: "a", title: "Song", artist: nil, key: nil, durationSec: 10)
        #expect(MiniPlayer.subtitle(none).isEmpty)
    }
}
