import Foundation
import MusicDecodedCore
import Observation
import os
import SwiftData

private let logger = Logger(subsystem: "com.musicdecoded.app", category: "Sync")

/// Which account last synced on this device, and its pull cursor. Not secrets, so UserDefaults.
struct SyncState {
    let defaults: UserDefaults

    private static let userIdKey = "sync.userId"
    private static let cursorKey = "sync.cursor"

    var userId: String? { defaults.string(forKey: Self.userIdKey) }
    var cursor: String? { defaults.string(forKey: Self.cursorKey) }

    func save(userId: String, cursor: String?) {
        defaults.set(userId, forKey: Self.userIdKey)
        if let cursor {
            defaults.set(cursor, forKey: Self.cursorKey)
        } else {
            defaults.removeObject(forKey: Self.cursorKey)
        }
    }
}

/// `date-time` strings from the API (`Z` or an explicit offset, with or without fractions).
enum SyncTime {
    struct InvalidDate: Error {
        let text: String
    }

    // ISO8601DateFormatter is thread-safe.
    nonisolated(unsafe) private static let fractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    nonisolated(unsafe) private static let whole: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static func format(_ date: Date) -> String {
        fractional.string(from: date)
    }

    static func parse(_ text: String) throws -> Date {
        if let date = fractional.date(from: text) ?? whole.date(from: text) {
            return date
        }
        throw InvalidDate(text: text)
    }
}

/// Library sync for accounts (T-404i, PLAN.md §5.2). Guests never sync.
/// 1. Push rows changed since their last sync (finished rows and tombstones) in batches of 500.
/// 2. Pull from the stored cursor until a page has fewer than 500 items.
/// 3. Merge last-write-wins on `updatedAt` (a tie keeps the local row).
/// 4. Fetch what synced rows are missing here: the analysis file, and the user's own audio
///    (file/recording with an `audioUploadId`). Streaming audio is never synced (R-1).
/// Ports apps/android/.../library/SyncRepository.kt and SyncWorker.kt.
@MainActor
@Observable
final class SyncService {
    /// Contract page size for both pull and push.
    static let page = 500
    static let downloadFailedMessage = "Couldn't download this transcription. Try again."
    /// Attempts per `requestSync()` (the first try plus retries with backoff).
    static let maxAttempts = 5

    let state: SyncState
    /// Where synced audio is saved (`<Application Support>/audio`).
    let audioDirectory: URL
    let analysisStore: AnalysisStore

    private let session: SessionStore
    private let context: ModelContext
    private let retryDelay: Duration
    /// The sync in flight; the next one waits for it (one sync at a time).
    @ObservationIgnored private var running: Task<Bool, any Error>? = nil
    /// The latest `requestSync()` run, with its retries.
    @ObservationIgnored private var scheduled: Task<Void, Never>? = nil

    init(
        session: SessionStore,
        context: ModelContext,
        analysisStore: AnalysisStore,
        audioDirectory: URL,
        defaults: UserDefaults = .standard,
        retryDelay: Duration = .seconds(30)
    ) {
        self.session = session
        self.context = context
        self.analysisStore = analysisStore
        self.audioDirectory = audioDirectory
        self.state = SyncState(defaults: defaults)
        self.retryDelay = retryDelay
    }

    // MARK: - Triggers

    /// Requests a sync: app foreground, after a job completes, after login (PLAN.md §5.2). The
    /// latest request replaces a pending one; failures retry with exponential backoff. Guests
    /// return right away.
    func requestSync() {
        scheduled?.cancel()
        scheduled = Task {
            await self.runWithRetries()
        }
    }

    private func runWithRetries() async {
        var delay = retryDelay
        for attempt in 1...Self.maxAttempts {
            if Task.isCancelled { return }
            do {
                try await sync()
                return
            } catch {
                logger.warning("Sync failed (attempt \(attempt)): \(String(describing: error), privacy: .public)")
                if attempt == Self.maxAttempts { return }
                try? await Task.sleep(for: delay)
                delay = delay * 2
            }
        }
    }

    // MARK: - Sync

    /// Runs one full sync. Returns `false` (and does nothing) for guests. Network errors are thrown.
    @discardableResult
    func sync() async throws -> Bool {
        let previous = running
        let task = Task { () async throws -> Bool in
            _ = try? await previous?.value
            return try await self.performSync()
        }
        running = task
        return try await task.value
    }

    private func performSync() async throws -> Bool {
        guard let account = session.account, let api = session.api else { return false }
        // First sync for this account here: full pull. Rows another account already synced aren't
        // re-pushed (only never-synced rows, i.e. guest data, and later edits are).
        if state.userId != account.userId {
            state.save(userId: account.userId, cursor: nil)
        }
        try await push(api, token: account.token)
        try await pull(api, token: account.token, userId: account.userId)
        try await fetchMissing(api, token: account.token)
        return true
    }

    private func push(_ api: APIClient, token: String) async throws {
        while true {
            let batch = pendingPush(limit: Self.page)
            if batch.isEmpty { return }
            let sentAt = batch.map(\.updatedAt)
            let items = batch.map(\.syncItem)
            let response: SyncItems
            do {
                response = try await api.pushLibrary(items, token: token)
            } catch let error as APIError where error.status == 400 {
                // The whole push is rejected if one audioUploadId isn't this account's (e.g. rows
                // left by another account that logged out here). The rows still sync, without audio.
                let withoutAudio = items.map { (item: SyncItem) -> SyncItem in
                    var copy = item
                    copy.audioUploadId = nil
                    return copy
                }
                response = try await api.pushLibrary(withoutAudio, token: token)
            }
            // Marks each row synced, unless it changed again while the push was in flight.
            for (item, updatedAt) in zip(batch, sentAt) where item.updatedAt == updatedAt {
                item.syncedAt = updatedAt
            }
            for remote in response.items {
                try merge(remote)
            }
            save()
            if batch.count < Self.page { return }
        }
    }

    private func pull(_ api: APIClient, token: String, userId: String) async throws {
        var cursor = state.cursor
        while true {
            let page = try await api.pullLibrary(since: cursor, token: token)
            for remote in page.items {
                try merge(remote)
            }
            save()
            if page.cursor.isEmpty == false {
                cursor = page.cursor
            }
            state.save(userId: userId, cursor: cursor)
            if page.items.count < Self.page { return }
        }
    }

    /// Applies `remote` if it's newer than the local row; local-only fields are kept.
    private func merge(_ remote: SyncItem) throws {
        let updatedAt = try SyncTime.parse(remote.updatedAt)
        let createdAt = try SyncTime.parse(remote.createdAt)
        let lastPlayedAt = try remote.lastPlayedAt.map { try SyncTime.parse($0) }
        let item: LibraryItem
        if let local = fetch(remote.trackId) {
            if updatedAt <= local.updatedAt { return }
            item = local
        } else {
            // trackIds become file names; skip anything unsafe.
            guard Transcription.isSafeTrackId(remote.trackId) else { return }
            item = LibraryItem(
                trackId: remote.trackId,
                title: remote.title,
                source: remote.source,
                status: .ready,
                createdAt: createdAt
            )
            context.insert(item)
        }
        item.title = remote.title
        item.artist = remote.artist
        item.source = remote.source
        item.sourceRef = remote.sourceRef
        item.durationSec = remote.durationSec ?? item.durationSec
        item.thumbnailUrl = remote.thumbnailUrl
        item.audioUploadId = remote.audioUploadId
        item.favorite = remote.favorite
        item.lastPlayedAt = lastPlayedAt
        item.createdAt = createdAt
        item.updatedAt = updatedAt
        item.tombstoned = remote.deleted
        item.syncedAt = updatedAt
    }

    /// Downloads missing analysis files and owned audio. One failure doesn't stop the others.
    private func fetchMissing(_ api: APIClient, token: String) async throws {
        for item in syncedVisible() {
            let trackId = item.trackId
            let wantsAnalysis = item.status == .ready || item.error == Self.downloadFailedMessage
            if wantsAnalysis, analysisStore.contains(trackId: trackId) == false {
                try await fetchAnalysis(api, token: token, trackId: trackId)
            }
            if item.source == .file || item.source == .recording,
               let uploadId = item.audioUploadId,
               hasLocalAudio(item) == false {
                try await fetchAudio(api, token: token, trackId: trackId, uploadId: uploadId)
            }
        }
    }

    private func fetchAnalysis(_ api: APIClient, token: String, trackId: String) async throws {
        var analysis: Analysis?
        do {
            let location = try await api.analysisLocation(trackId: trackId, token: token)
            let data = try await api.download(location.url)
            analysis = try analysisStore.save(data, for: trackId)
        } catch let error as CancellationError {
            throw error
        } catch {
            logger.warning("Analysis download failed for \(trackId, privacy: .public): \(String(describing: error), privacy: .public)")
        }
        guard let item = fetch(trackId) else { return }
        if let analysis {
            // Local metadata only: keep updatedAt so this doesn't count as a user change.
            item.apply(analysis, at: item.updatedAt)
        } else {
            // Shown as failed; the next sync tries again.
            item.status = .failed
            item.error = Self.downloadFailedMessage
        }
        save()
    }

    private func fetchAudio(_ api: APIClient, token: String, trackId: String, uploadId: String) async throws {
        guard Transcription.isSafeTrackId(trackId) else { return }
        do {
            let location = try await api.uploadLocation(id: uploadId, token: token)
            try FileManager.default.createDirectory(at: audioDirectory, withIntermediateDirectories: true)
            let fileName = "\(trackId).\(Self.fileExtension(for: location.contentType))"
            let target = audioDirectory.appending(path: fileName, directoryHint: .notDirectory)
            try await api.download(location.url, to: target)
            guard let item = fetch(trackId) else { return }
            item.localAudioPath = fileName
            save()
        } catch let error as CancellationError {
            throw error
        } catch {
            logger.warning("Audio download failed for \(trackId, privacy: .public): \(String(describing: error), privacy: .public)")
        }
    }

    // MARK: - Queries

    /// Push candidates: changed since their last sync, oldest change first. Only finished rows and
    /// tombstones; a row still analyzing is pushed once its job completes. Filtered in memory
    /// because comparing two dates (one optional) isn't a safe SwiftData predicate.
    func pendingPush(limit: Int) -> [LibraryItem] {
        let pending = allItems().filter { item in
            let changed = item.syncedAt.map { item.updatedAt > $0 } ?? true
            return changed && (item.status == .ready || item.tombstoned)
        }
        let sorted = pending.sorted { lhs, rhs in
            (lhs.updatedAt, lhs.trackId) < (rhs.updatedAt, rhs.trackId)
        }
        return Array(sorted.prefix(limit))
    }

    /// Finished local rows never synced: the guest data the migration screen lists (DESIGN.md §3.10).
    func unsyncedItems() -> [LibraryItem] {
        allItems().filter { item in
            item.syncedAt == nil && item.tombstoned == false && item.status == .ready
        }
    }

    /// Visible rows that went through sync (they may be missing their analysis or audio here).
    private func syncedVisible() -> [LibraryItem] {
        allItems().filter { item in
            item.syncedAt != nil && item.tombstoned == false && item.status != .analyzing
        }
    }

    private func allItems() -> [LibraryItem] {
        (try? context.fetch(FetchDescriptor<LibraryItem>())) ?? []
    }

    private func fetch(_ trackId: String) -> LibraryItem? {
        try? context.fetch(LibraryItem.withTrackId(trackId)).first
    }

    private func hasLocalAudio(_ item: LibraryItem) -> Bool {
        guard let fileName = item.localAudioPath else { return false }
        let url = audioDirectory.appending(path: fileName, directoryHint: .notDirectory)
        return FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
    }

    private func save() {
        try? context.save()
    }

    static func fileExtension(for contentType: String) -> String {
        let base = contentType.split(separator: ";").first.map(String.init) ?? contentType
        switch base.trimmingCharacters(in: .whitespaces).lowercased() {
        case "audio/mpeg": return "mp3"
        case "audio/mp4": return "m4a"
        case "audio/aac": return "aac"
        case "audio/wav", "audio/x-wav": return "wav"
        case "audio/flac": return "flac"
        case "audio/ogg": return "ogg"
        default: return "audio"
        }
    }
}

#if DEBUG
extension SyncService {
    /// Kept alive for the life of the preview process (a context needs its container).
    private static let previewContainer: ModelContainer = {
        do {
            return try ModelContainer(for: LibraryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        } catch {
            fatalError("Preview container: \(error)")
        }
    }()

    /// For SwiftUI previews: a guest session (never syncs), files in a temporary folder.
    static func preview() -> SyncService {
        let root = FileManager.default.temporaryDirectory.appending(path: "preview", directoryHint: .isDirectory)
        return SyncService(
            session: SessionStore.preview(),
            context: previewContainer.mainContext,
            analysisStore: AnalysisStore(directory: root.appending(path: "analysis", directoryHint: .isDirectory)),
            audioDirectory: root.appending(path: "audio", directoryHint: .isDirectory)
        )
    }
}
#endif

extension LibraryItem {
    /// The contract `LibraryItem` for `PUT /sync/library`.
    var syncItem: SyncItem {
        SyncItem(
            trackId: trackId,
            title: title,
            artist: artist,
            source: source,
            sourceRef: sourceRef,
            durationSec: durationSec,
            thumbnailUrl: thumbnailUrl,
            audioUploadId: audioUploadId,
            favorite: favorite,
            lastPlayedAt: lastPlayedAt.map(SyncTime.format),
            createdAt: SyncTime.format(createdAt),
            updatedAt: SyncTime.format(updatedAt),
            deleted: tombstoned
        )
    }
}
