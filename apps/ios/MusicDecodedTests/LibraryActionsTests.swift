import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

/// Row and Player ⋯ actions (T-213i): deleting the playing song clears the mini player.
@MainActor
struct LibraryActionsTests {
    let container: ModelContainer
    let nowPlaying = NowPlaying()
    let track: PlayerTrack

    init() throws {
        container = try ModelContainer(for: LibraryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        track = PlayerTrack(analysis: try PlayerFixtures.synthetic(), title: "Song", artist: nil, favorite: false, audioURL: nil)
    }

    private func load(_ trackId: String) -> any PlaybackEngine {
        nowPlaying.engine(for: trackId, track: track) { SilentPlaybackEngine(durationSec: $0.analysis.meta.durationSec) }
    }

    @Test func deletingThePlayingSongClearsTheMiniPlayer() throws {
        let item = LibraryItem(trackId: "a", title: "Song", source: .file, status: .ready)
        container.mainContext.insert(item)
        let engine = load("a")
        nowPlaying.togglePlay()
        #expect(nowPlaying.isPlaying)

        LibraryActions.delete(item, nowPlaying: nowPlaying, context: container.mainContext)

        #expect(nowPlaying.track == nil)
        #expect(nowPlaying.isPlaying == false)
        #expect(engine.isPlaying == false)
        #expect(item.tombstoned)
    }

    @Test func deletingAnotherSongKeepsPlaying() throws {
        let other = LibraryItem(trackId: "b", title: "Other", source: .file, status: .ready)
        container.mainContext.insert(other)
        _ = load("a")
        nowPlaying.togglePlay()

        LibraryActions.delete(other, nowPlaying: nowPlaying, context: container.mainContext)

        #expect(nowPlaying.track?.trackId == "a")
        #expect(nowPlaying.isPlaying)
        #expect(other.tombstoned)
    }

    @Test func favoriteBumpsUpdatedAt() throws {
        let item = LibraryItem(trackId: "a", title: "Song", source: .file, status: .ready, createdAt: .distantPast)
        container.mainContext.insert(item)
        LibraryActions.setFavorite(item, true, context: container.mainContext)
        #expect(item.favorite)
        #expect(item.updatedAt > .distantPast)
    }

    @Test(arguments: [("Sunny Days", "S"), ("  (intro) take", "I"), ("4 AM", "4"), ("", "♪"), ("...", "♪"), ("élan", "É")])
    func initialArtLetter(_ title: String, _ expected: String) {
        #expect(InitialArt.initial(of: title) == expected)
    }
}
