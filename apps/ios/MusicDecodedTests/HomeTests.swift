import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

/// Home rows and actions. Ports apps/android/.../ui/home/HomeViewModelTest.kt.
@MainActor
@Suite(.serialized)
struct HomeTests {
    let container: ModelContainer
    let context: ModelContext

    init() throws {
        container = try ModelContainer(
            for: LibraryItem.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        context = container.mainContext
    }

    private func item(
        _ id: String,
        status: LibraryItem.Status = .ready,
        local: Bool = true,
        createdAt: Date = Date(timeIntervalSince1970: 1_000_000)
    ) -> LibraryItem {
        let item = LibraryItem(trackId: id, title: "Song \(id)", artist: "Artist", source: .file, durationSec: 263.4, status: status, createdAt: createdAt)
        item.key = "F# minor"
        item.bpm = 136.1
        item.localAudioPath = local ? "\(id).m4a" : nil
        return item
    }

    private func job(_ trackId: String, progress: Int) -> Job {
        Job(jobId: "j1", trackId: trackId, status: .processing, stage: "detecting_beats", progress: progress, error: nil)
    }

    @Test func emptyHistoryHasNoRecentRows() throws {
        #expect(try context.fetch(LibraryItem.recent(limit: 3)).isEmpty)
    }

    @Test func recentIsAtMostThreeNewestVisibleRows() throws {
        let start = Date(timeIntervalSince1970: 1_000_000)
        for (offset, id) in ["d", "c", "b", "a"].enumerated() {
            context.insert(item(id, createdAt: start.addingTimeInterval(Double(offset))))
        }
        let hidden = item("z", createdAt: start.addingTimeInterval(100))
        hidden.markDeleted()
        context.insert(hidden)
        try context.save()

        #expect(try context.fetch(LibraryItem.recent(limit: 3)).map(\.trackId) == ["a", "b", "c"])
    }

    @Test func rowShowsMetaOfflineAndStatus() {
        let ready = SongRowState(item("a"), job: nil)
        #expect(ready.meta == "4:23 · F♯ minor · 136 BPM")
        #expect(ready.isOffline)
        #expect(ready.status == .ready)

        let failed = SongRowState(item("b", status: .failed, local: false), job: nil)
        #expect(failed.status == .failed)
        #expect(failed.isOffline == false)
    }

    @Test func analyzingRowTakesProgressFromTheJob() {
        let analyzing = item("a", status: .analyzing)
        #expect(SongRowState(analyzing, job: nil).status == .analyzing(progress: nil))
        #expect(SongRowState(analyzing, job: job("a", progress: 42)).status == .analyzing(progress: 42))
        #expect(SongRowState(analyzing, job: job("a", progress: 140)).status == .analyzing(progress: 100))
    }

    @Test func favoriteBumpsUpdatedAt() throws {
        let row = item("a")
        context.insert(row)
        let later = row.updatedAt.addingTimeInterval(60)
        row.setFavorite(true, at: later)
        try context.save()

        let fetched = try #require(try context.fetch(LibraryItem.withTrackId("a")).first)
        #expect(fetched.favorite)
        #expect(fetched.updatedAt == later)
    }

    @Test func metaSkipsUnknownPartsAndUsesGlyphs() {
        #expect(SongMetadata.line(durationSec: nil, key: nil, bpm: nil) == "")
        #expect(SongMetadata.line(durationSec: nil, key: "Bb major", bpm: nil) == "B♭ major")
        #expect(SongMetadata.displayKey("D major") == "D major")
    }
}
