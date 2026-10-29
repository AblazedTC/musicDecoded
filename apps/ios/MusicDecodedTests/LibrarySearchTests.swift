import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

/// Library search, filters and counts against an in-memory store.
/// Ports apps/android/.../ui/library/LibraryViewModelTest.kt.
@MainActor
@Suite(.serialized)
struct LibrarySearchTests {
    let container: ModelContainer
    let context: ModelContext

    init() throws {
        container = try ModelContainer(
            for: LibraryItem.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        context = container.mainContext
        try seed()
    }

    /// newest → oldest: rec (recording, F# minor), fav (favorite, Bb major), local (offline), gone (deleted)
    private func seed() throws {
        let start = Date(timeIntervalSince1970: 1_000_000)
        let local = LibraryItem(trackId: "local", title: "Sunny Days", artist: "The Beams", source: .file, status: .ready, createdAt: start)
        local.localAudioPath = "local.m4a"
        local.key = "D major"
        let fav = LibraryItem(trackId: "fav", title: "Somehow", artist: "Day Six", source: .file, status: .ready, createdAt: start.addingTimeInterval(1))
        fav.favorite = true
        fav.key = "Bb major"
        let rec = LibraryItem(trackId: "rec", title: "Voice memo", source: .recording, status: .analyzing, createdAt: start.addingTimeInterval(2))
        rec.key = "F# minor"
        rec.localAudioPath = "rec.m4a"
        let gone = LibraryItem(trackId: "gone", title: "Sunny Gone", source: .file, status: .ready, createdAt: start.addingTimeInterval(3))
        gone.markDeleted()
        for item in [local, fav, rec, gone] {
            context.insert(item)
        }
        try context.save()
    }

    private func ids(_ text: String, _ filter: LibraryFilter = .all) throws -> [String] {
        try context.fetch(LibraryItem.search(LibraryQuery(text: text, filter: filter))).map(\.trackId)
    }

    @Test func queryIsTrimmedAndMappedToStoredAccidentals() {
        #expect(LibraryQuery(text: "  F♯ min ", filter: .all).text == "F# min")
        #expect(LibraryQuery(text: "B♭", filter: .all).text == "Bb")
        #expect(LibraryQuery(text: " ", filter: .all).isUnfiltered)
        #expect(LibraryQuery(text: "", filter: .favorites).isUnfiltered == false)
        #expect(LibraryQuery(text: "a ", filter: .all) == LibraryQuery(text: "a", filter: .all))
    }

    @Test func allSongsNewestFirstWithoutTombstones() throws {
        #expect(try ids("") == ["rec", "fav", "local"])
    }

    @Test func searchMatchesTitleArtistAndKey() throws {
        #expect(try ids("sunny") == ["local"])
        #expect(try ids("day six") == ["fav"])
        #expect(try ids("F♯") == ["rec"])
        #expect(try ids("B♭ major") == ["fav"])
        #expect(try ids("zzz").isEmpty)
    }

    @Test func filters() throws {
        #expect(try ids("", .favorites) == ["fav"])
        #expect(try ids("", .offline) == ["rec", "local"])
        #expect(try ids("", .recordings) == ["rec"])
        #expect(try ids("memo", .offline) == ["rec"])
        #expect(try ids("sunny", .favorites).isEmpty)
    }

    @Test func counts() throws {
        #expect(try context.fetchCount(LibraryItem.decodedItems) == 2)
        #expect(try context.fetchCount(LibraryItem.offlineItems) == 2)
    }
}
