import Foundation
import MusicDecodedCore
import os
import SwiftData
import Testing
@testable import MusicDecoded

private let audioBytes = Data("fake-mp3-bytes".utf8)

private struct UnexpectedRequest: Error {
    let route: String
}

/// An in-memory `/sync/library` with the server rules from T-402: LWW on updatedAt (ties keep the
/// server), pulls ordered by a server sequence, 500 per page, the cursor = last sequence seen.
/// Ports FakeSyncServer from apps/android/.../library/SyncRepositoryTest.kt.
final class FakeSyncServer: Sendable {
    private struct Row: Sendable {
        var item: SyncItem
        var seq: Int
    }

    private struct State: Sendable {
        var rows: [String: Row] = [:]
        var seq = 0
        var analysisAvailable = true
        /// Rejects (400) any push that carries an audioUploadId, like a foreign upload would.
        var rejectAudioUploadIds = false
        var calls: [String] = []
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let fixture: Data

    init(fixture: Data) {
        self.fixture = fixture
    }

    var analysisAvailable: Bool {
        get { state.withLock { $0.analysisAvailable } }
        set { state.withLock { $0.analysisAvailable = newValue } }
    }

    var rejectAudioUploadIds: Bool {
        get { state.withLock { $0.rejectAudioUploadIds } }
        set { state.withLock { $0.rejectAudioUploadIds = newValue } }
    }

    var calls: [String] { state.withLock { $0.calls } }

    func clearCalls() {
        state.withLock { $0.calls.removeAll() }
    }

    func items() -> [SyncItem] {
        state.withLock { state in state.rows.values.sorted { $0.seq < $1.seq }.map(\.item) }
    }

    func item(_ trackId: String) -> SyncItem? {
        state.withLock { $0.rows[trackId]?.item }
    }

    /// Writes as if another device had pushed `item`.
    func put(_ item: SyncItem) {
        state.withLock { Self.put(item, into: &$0) }
    }

    /// A client for one device. Every device talks to the same rows.
    func client() -> APIClient {
        StubURLProtocol.client { request in
            try self.handle(request)
        }
    }

    private static func put(_ item: SyncItem, into state: inout State) {
        let newTime = (try? SyncTime.parse(item.updatedAt)) ?? .distantPast
        if let existing = state.rows[item.trackId] {
            let oldTime = (try? SyncTime.parse(existing.item.updatedAt)) ?? .distantPast
            guard newTime > oldTime else { return }
        }
        state.seq += 1
        state.rows[item.trackId] = Row(item: item, seq: state.seq)
    }

    private func handle(_ request: URLRequest) throws -> (status: Int, body: Data) {
        let method = request.httpMethod ?? "GET"
        let path = request.url?.path() ?? ""
        let host = request.url?.host() ?? ""
        let route = "\(method) \(path)"
        state.withLock { $0.calls.append(route) }

        if method == "GET", path == "/sync/library" {
            let components = request.url.flatMap { URLComponents(url: $0, resolvingAgainstBaseURL: false) }
            let since = components?.queryItems?.first { $0.name == "since" }?.value.flatMap { Int($0) } ?? 0
            let page = state.withLock { (state: inout State) -> SyncPage in
                let rows = state.rows.values.filter { $0.seq > since }.sorted { $0.seq < $1.seq }.prefix(500)
                let cursor = rows.last.map { String($0.seq) } ?? String(since)
                return SyncPage(items: rows.map(\.item), cursor: cursor)
            }
            return (200, try JSONEncoder().encode(page))
        }
        if method == "PUT", path == "/sync/library" {
            let pushed = try JSONDecoder().decode(SyncItems.self, from: request.bodyData ?? Data()).items
            guard pushed.count <= 500 else { throw UnexpectedRequest(route: "push over the contract limit") }
            let rejected = state.withLock { $0.rejectAudioUploadIds } && pushed.contains { $0.audioUploadId != nil }
            if rejected {
                return (400, Data(#"{"code": "bad_request", "message": "Unknown upload"}"#.utf8))
            }
            let result = state.withLock { (state: inout State) -> SyncItems in
                for item in pushed {
                    Self.put(item, into: &state)
                }
                return SyncItems(items: pushed.compactMap { state.rows[$0.trackId]?.item })
            }
            return (200, try JSONEncoder().encode(result))
        }
        if method == "GET", path.hasPrefix("/tracks/"), path.hasSuffix("/analysis") {
            let trackId = String(path.dropFirst("/tracks/".count).dropLast("/analysis".count))
            guard analysisAvailable else { return (404, SampleJSON.notFound) }
            return (200, Data(#"{"url": "https://\#(host)/analysis/\#(trackId).json", "schemaVersion": 1}"#.utf8))
        }
        if method == "GET", path.hasPrefix("/analysis/") {
            return (200, fixture)
        }
        if method == "GET", path.hasPrefix("/uploads/") {
            let id = String(path.dropFirst("/uploads/".count))
            return (200, Data(#"{"url": "https://\#(host)/blob/\#(id)", "contentType": "audio/mpeg", "expiresAt": "2030-01-01T00:00:00Z"}"#.utf8))
        }
        if method == "GET", path.hasPrefix("/blob/") {
            return (200, audioBytes)
        }
        throw UnexpectedRequest(route: route)
    }
}

/// One phone: its own SwiftData store, files and sync state, talking to the shared fake server.
@MainActor
final class SyncDevice {
    let container: ModelContainer
    let analysisStore: AnalysisStore
    let audioDirectory: URL
    let sync: SyncService

    var context: ModelContext { container.mainContext }

    init(server: FakeSyncServer, root: URL, kind: AuthResponse.Kind = .account) throws {
        let base = root.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        let container = try ModelContainer(
            for: LibraryItem.self,
            configurations: ModelConfiguration(UUID().uuidString, isStoredInMemoryOnly: true)
        )
        let analysisStore = try AnalysisStore(directory: AppDirectories.analysis(in: base))
        let audioDirectory = try AppDirectories.audio(in: base)
        self.container = container
        self.analysisStore = analysisStore
        self.audioDirectory = audioDirectory
        let defaults = UserDefaults(suiteName: "SyncServiceTests-\(UUID().uuidString)")!
        let session = SessionStore(
            api: server.client(),
            secureStore: InMemorySecureStore([
                "deviceId": "device", "token": "token", "kind": kind.rawValue, "userId": "user-1",
            ]),
            defaults: defaults
        )
        sync = SyncService(
            session: session,
            context: container.mainContext,
            analysisStore: analysisStore,
            audioDirectory: audioDirectory,
            defaults: defaults
        )
    }

    func insert(_ item: LibraryItem) {
        context.insert(item)
        try? context.save()
    }

    func item(_ trackId: String) -> LibraryItem? {
        try? context.fetch(LibraryItem.withTrackId(trackId)).first
    }

    func count() -> Int {
        (try? context.fetchCount(FetchDescriptor<LibraryItem>())) ?? 0
    }

    /// The synced fields of every server row (duration is refreshed locally from the analysis).
    func snapshot(of server: FakeSyncServer) -> [String: SyncItem] {
        var result: [String: SyncItem] = [:]
        for remote in server.items() {
            guard var local = item(remote.trackId)?.syncItem else { continue }
            local.durationSec = nil
            result[remote.trackId] = local
        }
        return result
    }
}

private func at(_ seconds: TimeInterval) -> Date {
    Date(timeIntervalSince1970: 1_793_275_200 + seconds) // 2026-10-29T12:00:00Z
}

@MainActor
private func localItem(
    _ trackId: String,
    updatedAt: Date = at(0),
    source: Analysis.Source = .youtube,
    title: String? = nil,
    favorite: Bool = false,
    deleted: Bool = false,
    audioUploadId: String? = nil
) -> LibraryItem {
    let item = LibraryItem(trackId: trackId, title: title ?? "Song \(trackId)", source: source, status: .ready, createdAt: at(0))
    item.favorite = favorite
    item.updatedAt = updatedAt
    item.tombstoned = deleted
    item.audioUploadId = audioUploadId
    return item
}

private func remoteItem(
    _ trackId: String,
    updatedAt: Date = at(0),
    source: Analysis.Source = .youtube,
    title: String? = nil,
    favorite: Bool = false,
    deleted: Bool = false,
    audioUploadId: String? = nil
) -> SyncItem {
    SyncItem(
        trackId: trackId,
        title: title ?? "Song \(trackId)",
        source: source,
        audioUploadId: audioUploadId,
        favorite: favorite,
        createdAt: SyncTime.format(at(0)),
        updatedAt: SyncTime.format(updatedAt),
        deleted: deleted
    )
}

/// Ports apps/android/.../library/SyncRepositoryTest.kt.
@MainActor
@Suite(.serialized)
struct SyncServiceTests {
    private final class BundleToken {}

    let root: URL
    let server: FakeSyncServer

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appending(path: "SyncServiceTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        let url = try #require(Bundle(for: BundleToken.self).url(forResource: "sample.analysis", withExtension: "json"))
        server = FakeSyncServer(fixture: try Data(contentsOf: url))
    }

    private func device(_ kind: AuthResponse.Kind = .account) throws -> SyncDevice {
        try SyncDevice(server: server, root: root, kind: kind)
    }

    private func cleanUp() {
        try? FileManager.default.removeItem(at: root)
    }

    @Test func twoDevicesConverge() async throws {
        defer { cleanUp() }
        let a = try device()
        let b = try device()
        a.insert(localItem("t1", updatedAt: at(1), title: "From A"))
        b.insert(localItem("t2", updatedAt: at(2), title: "From B"))

        try await a.sync.sync()
        try await b.sync.sync()
        try await a.sync.sync()

        #expect(Set(server.items().map(\.trackId)) == ["t1", "t2"])
        #expect(a.snapshot(of: server) == b.snapshot(of: server))
        #expect(a.item("t2")?.title == "From B")
        // Pulled rows got their analysis file and are playable.
        #expect(a.analysisStore.contains(trackId: "t2"))
        #expect(a.item("t2")?.status == .ready)
        #expect(b.analysisStore.contains(trackId: "t1"))
    }

    @Test func tombstonesPropagate() async throws {
        defer { cleanUp() }
        let a = try device()
        let b = try device()
        a.insert(localItem("t1", updatedAt: at(1)))
        try await a.sync.sync()
        try await b.sync.sync()
        #expect(b.item("t1")?.tombstoned == false)

        a.item("t1")?.markDeleted(at: at(5))
        try a.context.save()
        try await a.sync.sync()
        try await b.sync.sync()

        #expect(server.item("t1")?.deleted == true)
        #expect(b.item("t1")?.tombstoned == true)
    }

    @Test func newerRemoteWinsOlderRemoteLoses() async throws {
        defer { cleanUp() }
        let a = try device()
        a.insert(localItem("newer", updatedAt: at(10)))
        a.insert(localItem("older", updatedAt: at(10)))
        try await a.sync.sync()

        // Another device: a newer edit of one row, an older edit of the other.
        server.put(remoteItem("newer", updatedAt: at(20), favorite: true))
        server.put(remoteItem("older", updatedAt: at(5), favorite: true)) // rejected by the server too
        // And a local edit, newer than the server, made offline.
        a.item("older")?.setFavorite(true, at: at(30))
        try a.context.save()
        try await a.sync.sync()

        #expect(a.item("newer")?.favorite == true)
        #expect(a.item("newer")?.updatedAt == at(20))
        let older = try #require(server.item("older"))
        #expect(try SyncTime.parse(older.updatedAt) == at(30))
        #expect(older.favorite)
    }

    @Test func localRowNewerThanAPulledOneIsKept() async throws {
        defer { cleanUp() }
        let a = try device()
        server.put(remoteItem("t1", updatedAt: at(5), title: "Server title"))
        let local = localItem("t1", updatedAt: at(9), title: "Local title")
        local.syncedAt = at(9)
        a.insert(local)

        try await a.sync.sync()

        #expect(a.item("t1")?.title == "Local title")
    }

    @Test func pullPaginatesUntilAShortPageAndStoresTheCursor() async throws {
        defer { cleanUp() }
        for index in 1...1201 {
            server.put(remoteItem("t\(index)", updatedAt: at(TimeInterval(index)), deleted: true))
        }
        let a = try device()

        try await a.sync.sync()

        #expect(server.calls.filter { $0 == "GET /sync/library" }.count == 3)
        #expect(a.count() == 1201)
        #expect(a.sync.state.cursor == "1201")

        // Next sync starts from the cursor: one empty page.
        server.clearCalls()
        try await a.sync.sync()
        #expect(server.calls == ["GET /sync/library"])
    }

    @Test func pushBatchesOf500() async throws {
        defer { cleanUp() }
        server.analysisAvailable = false
        let a = try device()
        for index in 1...501 {
            a.context.insert(localItem("t\(index)", updatedAt: at(TimeInterval(index))))
        }
        try a.context.save()

        try await a.sync.sync()

        #expect(server.calls.filter { $0 == "PUT /sync/library" }.count == 2)
        #expect(server.items().count == 501)
        #expect(a.sync.pendingPush(limit: 10).isEmpty)
    }

    @Test func aRejectedPushIsRetriedOnceWithoutAudioUploadIds() async throws {
        defer { cleanUp() }
        server.rejectAudioUploadIds = true
        let a = try device()
        a.insert(localItem("rec", updatedAt: at(1), source: .recording, audioUploadId: "someone-elses"))

        try await a.sync.sync()

        #expect(server.calls.filter { $0 == "PUT /sync/library" }.count == 2)
        let pushed = try #require(server.item("rec"))
        #expect(pushed.audioUploadId == nil)
        #expect(a.sync.pendingPush(limit: 10).isEmpty)
    }

    @Test func ownAudioIsDownloadedStreamingAudioIsNot() async throws {
        defer { cleanUp() }
        server.put(remoteItem("rec", updatedAt: at(1), source: .recording, audioUploadId: "up-1"))
        server.put(remoteItem("yt", updatedAt: at(1), source: .youtube))
        let a = try device()

        try await a.sync.sync()

        let fileName = try #require(a.item("rec")?.localAudioPath)
        #expect(fileName == "rec.mp3")
        let saved = try Data(contentsOf: a.audioDirectory.appending(path: fileName))
        #expect(saved == audioBytes)
        #expect(a.item("yt")?.localAudioPath == nil)
        #expect(server.calls.filter { $0.hasPrefix("GET /uploads/") }.count == 1)
    }

    @Test func missingAnalysisMarksTheRowFailedAndRetriesNextSync() async throws {
        defer { cleanUp() }
        server.analysisAvailable = false
        server.put(remoteItem("t1", updatedAt: at(1)))
        let a = try device()

        try await a.sync.sync()
        #expect(a.item("t1")?.status == .failed)
        #expect(a.item("t1")?.error == SyncService.downloadFailedMessage)

        server.analysisAvailable = true
        try await a.sync.sync()
        #expect(a.item("t1")?.status == .ready)
        #expect(a.analysisStore.contains(trackId: "t1"))
    }

    @Test func guestDoesNothing() async throws {
        defer { cleanUp() }
        let guest = try device(.guest)
        guest.insert(localItem("t1", updatedAt: at(1)))

        let synced = try await guest.sync.sync()

        #expect(synced == false)
        #expect(server.calls.isEmpty)
    }

    @Test func timesRoundTrip() throws {
        #expect(try SyncTime.parse("2026-10-29T12:00:00Z") == at(0))
        let fractional = try SyncTime.parse("2026-10-29T12:00:01.500+00:00")
        #expect(abs(fractional.timeIntervalSince(at(1.5))) < 0.001)
        #expect(try SyncTime.parse("2026-10-29T14:00:00+02:00") == at(0))
        #expect(try SyncTime.parse(SyncTime.format(at(42))) == at(42))
        #expect(throws: SyncTime.InvalidDate.self) {
            try SyncTime.parse("yesterday")
        }
    }

    @Test func audioExtensions() {
        #expect(SyncService.fileExtension(for: "audio/mpeg") == "mp3")
        #expect(SyncService.fileExtension(for: "audio/mp4; codecs=mp4a") == "m4a")
        #expect(SyncService.fileExtension(for: "audio/x-wav") == "wav")
        #expect(SyncService.fileExtension(for: "application/octet-stream") == "audio")
    }
}
