import Foundation
import MusicDecodedCore
import Testing
@testable import MusicDecoded

/// AnalysisStore and the audio folder, in a fresh temporary directory per test.
struct AnalysisStoreTests {
    private final class BundleToken {}

    let root: URL
    let store: AnalysisStore

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appending(path: "AnalysisStoreTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        store = try AnalysisStore(directory: AppDirectories.analysis(in: root))
    }

    /// contracts/fixtures/sample.analysis.json, added to the test bundle by reference.
    private func fixtureData() throws -> Data {
        let bundle = Bundle(for: BundleToken.self)
        let url = try #require(bundle.url(forResource: "sample.analysis", withExtension: "json"))
        return try Data(contentsOf: url)
    }

    @Test func fixtureRoundTripsVerbatim() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let data = try fixtureData()
        let expected = try Analysis(data: data)
        let trackId = expected.meta.trackId

        let saved = try store.save(data, for: trackId)

        #expect(saved == expected)
        #expect(store.contains(trackId: trackId))
        #expect(try Data(contentsOf: store.fileURL(for: trackId)) == data)
        #expect(try store.load(trackId: trackId) == expected)
        #expect(try store.fileURL(for: trackId).lastPathComponent == "\(trackId).json")
        #expect(try store.fileURL(for: trackId).deletingLastPathComponent().lastPathComponent == "analysis")
    }

    @Test func missingFileThrows() {
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(store.contains(trackId: "missing") == false)
        #expect(throws: (any Error).self) {
            try store.load(trackId: "missing")
        }
    }

    @Test func invalidDataIsNotSaved() {
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(throws: (any Error).self) {
            try store.save(Data("not json".utf8), for: "track-1")
        }
        #expect(store.contains(trackId: "track-1") == false)
    }

    @Test(arguments: ["", "../escape", "a/b", ".hidden"])
    func unsafeTrackIdsAreRejected(trackId: String) {
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(throws: AnalysisStore.InvalidTrackID.self) {
            try store.fileURL(for: trackId)
        }
    }

    @Test func deleteRemovesTheFile() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let data = try fixtureData()
        try store.save(data, for: "track-1")

        try store.delete(trackId: "track-1")

        #expect(store.contains(trackId: "track-1") == false)
    }

    @Test func audioFolderIsExcludedFromBackup() throws {
        defer { try? FileManager.default.removeItem(at: root) }

        let audio = try AppDirectories.audio(in: root)

        // A fresh URL, so the value is read from disk rather than the URL's cache.
        let reloaded = URL(filePath: audio.path(percentEncoded: false), directoryHint: .isDirectory)
        let values = try reloaded.resourceValues(forKeys: [.isExcludedFromBackupKey])
        #expect(values.isExcludedFromBackup == true)
    }
}
