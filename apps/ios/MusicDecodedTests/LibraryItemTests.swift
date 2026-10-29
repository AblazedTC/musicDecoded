import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

/// LibraryItem CRUD against an in-memory store, on the main context.
@MainActor
@Suite(.serialized)
struct LibraryItemTests {
    let container: ModelContainer
    let context: ModelContext

    init() throws {
        container = try ModelContainer(
            for: LibraryItem.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        context = container.mainContext
    }

    private func insert(_ trackId: String, createdAt: Date) -> LibraryItem {
        let item = LibraryItem(trackId: trackId, title: "Song \(trackId)", source: .file, createdAt: createdAt)
        context.insert(item)
        return item
    }

    @Test func createAndReadNewestFirst() throws {
        let start = Date(timeIntervalSince1970: 1_000_000)
        _ = insert("old", createdAt: start)
        _ = insert("new", createdAt: start.addingTimeInterval(120))
        _ = insert("middle", createdAt: start.addingTimeInterval(60))
        try context.save()

        let items = try context.fetch(LibraryItem.newestFirst)

        #expect(items.map(\.trackId) == ["new", "middle", "old"])
    }

    @Test func newItemDefaults() throws {
        _ = insert("track-1", createdAt: .now)
        try context.save()

        let fetched = try #require(try context.fetch(LibraryItem.withTrackId("track-1")).first)
        #expect(fetched.source == .file)
        #expect(fetched.status == .analyzing)
        #expect(fetched.favorite == false)
        #expect(fetched.tombstoned == false)
        #expect(fetched.updatedAt == fetched.createdAt)
    }

    @Test func update() throws {
        let item = insert("track-1", createdAt: .now)
        try context.save()

        item.status = .ready
        item.jobId = nil
        item.key = "B minor"
        item.bpm = 140
        item.favorite = true
        item.localAudioPath = "track-1.m4a"
        try context.save()

        let fetched = try #require(try context.fetch(LibraryItem.withTrackId("track-1")).first)
        #expect(fetched.status == .ready)
        #expect(fetched.key == "B minor")
        #expect(fetched.bpm == 140)
        #expect(fetched.favorite)
        #expect(fetched.localAudioPath == "track-1.m4a")
    }

    @Test func deleteIsATombstone() throws {
        let created = Date(timeIntervalSince1970: 1_000_000)
        let item = insert("track-1", createdAt: created)
        _ = insert("track-2", createdAt: created)
        try context.save()

        let deletedAt = created.addingTimeInterval(60)
        item.markDeleted(at: deletedAt)
        try context.save()

        #expect(try context.fetch(LibraryItem.newestFirst).map(\.trackId) == ["track-2"])
        let tombstone = try #require(try context.fetch(LibraryItem.withTrackId("track-1")).first)
        #expect(tombstone.tombstoned)
        #expect(tombstone.updatedAt == deletedAt)
    }

    @Test func hardDeleteRemovesTheRow() throws {
        let item = insert("track-1", createdAt: .now)
        try context.save()

        context.delete(item)
        try context.save()

        #expect(try context.fetchCount(FetchDescriptor<LibraryItem>()) == 0)
    }
}
