import Foundation
import MusicDecodedCore
import Observation
import SwiftData

/// Asks the system for time to finish work after the app leaves the foreground. Returns the call
/// that ends it. The app passes `AppBackgroundActivity.begin`; tests pass nothing.
typealias BackgroundActivity = @MainActor (_ name: String) -> @MainActor () -> Void

/// The job runner every source uses (T-204i, PLAN.md §3.2):
/// SHA-256 → `POST /uploads` → PUT bytes → `POST /jobs` → LibraryItem `analyzing` →
/// poll `GET /jobs/{id}` → `GET /tracks/{id}/analysis` → download → `AnalysisStore.save` → `ready`.
/// A failed job sets `failed` with a message; `retry` reuses the upload.
///
/// Backgrounding: uploads run inside a background task (about 30 s of extra time). Polling is a
/// plain `Task` that iOS suspends with the app; the server keeps working regardless, and
/// `resumePending()` (launch and every return to the foreground) picks up every item that's still
/// `analyzing`. A background `URLSession` isn't used: it can't run the multi-step flow, and the
/// short status polls gain nothing from it.
@MainActor
@Observable
final class TranscriptionService {
    /// The openapi `GET /jobs/{jobId}` rule: every 2 s, backing off to 10 s after 60 s.
    struct PollTiming: Sendable {
        var fast: Duration = .seconds(2)
        var slow: Duration = .seconds(10)
        var backOffAfter: Duration = .seconds(60)
    }

    /// The last known `Job` per trackId while it runs (drives Processing screen progress, T-211).
    private(set) var jobs: [String: Job] = [:]

    /// Where audio lives: `<Application Support>/audio` (PLAN.md §5.1).
    let audioDirectory: URL

    private let session: SessionStore
    private let context: ModelContext
    private let analysisStore: AnalysisStore
    private let timing: PollTiming
    private let backgroundActivity: BackgroundActivity?

    private struct Poll {
        let id: UUID
        let task: Task<Void, Never>
    }

    @ObservationIgnored private var polls: [String: Poll] = [:]

    /// Consecutive network/server errors after which polling gives up and marks the item failed.
    private static let maxPollFailures = 8

    init(
        session: SessionStore,
        context: ModelContext,
        analysisStore: AnalysisStore,
        audioDirectory: URL,
        timing: PollTiming = PollTiming(),
        backgroundActivity: BackgroundActivity? = nil
    ) {
        self.session = session
        self.context = context
        self.analysisStore = analysisStore
        self.audioDirectory = audioDirectory
        self.timing = timing
        self.backgroundActivity = backgroundActivity
    }

    // MARK: - Files

    /// A fresh file URL in `audio/` for a source to write into before the trackId is known.
    func newPendingFileURL(extension fileExtension: String) throws -> URL {
        try FileManager.default.createDirectory(at: audioDirectory, withIntermediateDirectories: true)
        let name = "pending-\(UUID().uuidString.lowercased()).\(fileExtension.lowercased())"
        return audioDirectory.appending(path: name, directoryHint: .notDirectory)
    }

    /// The item's local audio file, if it has one.
    func audioURL(for item: LibraryItem) -> URL? {
        item.localAudioPath.map { audioDirectory.appending(path: $0, directoryHint: .notDirectory) }
    }

    // MARK: - Jobs

    /// Uploads `audio`, requests the analysis and starts polling. Returns the trackId as soon as the
    /// job exists (the LibraryItem is then `analyzing`). Throws `TranscriptionError`.
    @discardableResult
    func start(_ audio: LocalAudio) async throws -> String {
        let end = backgroundActivity?("Upload audio")
        defer { end?() }
        return try await userFacing { () async throws -> String in
            let uploadId = try await upload(audio.fileURL)
            let job = try await createJob(source: audio.source, uploadId: uploadId, title: audio.title)
            let fileName = try moveToTrack(audio.fileURL, trackId: job.trackId)

            let item: LibraryItem
            if let existing = fetch(job.trackId) {
                item = existing
            } else {
                item = LibraryItem(trackId: job.trackId, title: audio.title, source: audio.source)
                context.insert(item)
            }
            item.title = audio.title
            item.durationSec = audio.durationSec ?? item.durationSec
            item.audioUploadId = uploadId
            item.localAudioPath = fileName
            item.tombstoned = false
            begin(item, job)
            return job.trackId
        }
    }

    /// Requests a new analysis for a failed (or re-analyzed) item. Reuses its upload, and uploads
    /// the local copy again only if `POST /jobs` says the upload is gone (404).
    func retry(trackId: String) async throws {
        let end = backgroundActivity?("Retry analysis")
        defer { end?() }
        try await userFacing { () async throws -> Void in
            guard let item = fetch(trackId) else {
                throw TranscriptionError(message: Transcription.genericMessage)
            }
            do {
                var uploadId: String
                if let existing = item.audioUploadId {
                    uploadId = existing
                } else {
                    uploadId = try await upload(localAudioURL(item))
                }
                let job: Job
                do {
                    job = try await createJob(source: item.source, uploadId: uploadId, title: item.title)
                } catch let error as APIError where error.status == 404 {
                    uploadId = try await upload(localAudioURL(item))
                    job = try await createJob(source: item.source, uploadId: uploadId, title: item.title)
                }
                item.audioUploadId = uploadId
                begin(item, job)
            } catch let error as CancellationError {
                throw error
            } catch {
                markFailed(trackId, message: Transcription.userMessage(for: error))
                throw error
            }
        }
    }

    /// Starts polling every item that's still `analyzing` and isn't already being polled.
    /// Call at launch and whenever the app returns to the foreground.
    func resumePending() {
        let items = (try? context.fetch(LibraryItem.analyzingItems)) ?? []
        for item in items {
            if item.jobId == nil {
                // Interrupted before the job existed: nothing to poll.
                markFailed(item.trackId, message: Transcription.genericMessage)
            } else {
                schedulePolling(item.trackId, replace: false)
            }
        }
    }

    /// The running poll for `trackId`, if any (tests await it).
    func pollingTask(for trackId: String) -> Task<Void, Never>? {
        polls[trackId]?.task
    }

    func markFailed(_ trackId: String, message: String) {
        jobs[trackId] = nil
        guard let item = fetch(trackId) else { return }
        item.status = .failed
        item.jobId = nil
        item.error = message
        item.updatedAt = .now
        save()
    }

    // MARK: - Polling

    /// Polls the item's job until it ends. Network/server errors are thrown so the caller can try
    /// again; a terminal answer is saved.
    func poll(trackId: String) async throws {
        guard let item = fetch(trackId), item.status == .analyzing, let jobId = item.jobId else { return }
        var elapsed = Duration.zero
        while true {
            let job: Job
            do {
                job = try await session.authorized { api, token in
                    try await api.job(id: jobId, token: token)
                }
            } catch let error as APIError where error.status == 404 {
                markFailed(trackId, message: Transcription.expiredMessage)
                return
            }
            // A newer job replaced this one (retry): leave it to that poll.
            guard fetch(trackId)?.jobId == jobId else { return }
            jobs[trackId] = job

            switch job.status {
            case .complete:
                try await complete(trackId)
                return
            case .failed:
                let message = job.error?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                markFailed(trackId, message: message.isEmpty ? Transcription.analysisFailedMessage : message)
                return
            case .queued, .processing:
                let interval = elapsed < timing.backOffAfter ? timing.fast : timing.slow
                try await Task.sleep(for: interval)
                elapsed += interval
            }
        }
    }

    private func schedulePolling(_ trackId: String, replace: Bool) {
        if let existing = polls[trackId] {
            guard replace else { return }
            existing.task.cancel()
        }
        let id = UUID()
        let task = Task { [weak self] in
            guard let self else { return }
            await self.runPolling(trackId, id: id)
        }
        polls[trackId] = Poll(id: id, task: task)
    }

    /// `poll` with retries for transient errors (the iOS stand-in for WorkManager's backoff).
    private func runPolling(_ trackId: String, id: UUID) async {
        var failures = 0
        while !Task.isCancelled {
            do {
                try await poll(trackId: trackId)
                break
            } catch {
                if Task.isCancelled { break }
                failures += 1
                if failures >= Self.maxPollFailures {
                    markFailed(trackId, message: Transcription.userMessage(for: error))
                    break
                }
                try? await Task.sleep(for: timing.slow)
            }
        }
        if polls[trackId]?.id == id {
            polls[trackId] = nil
        }
    }

    /// Downloads and saves the analysis, then marks the item ready.
    private func complete(_ trackId: String) async throws {
        guard let client = session.api else { throw SessionStore.SessionError.missingBaseURL }
        let analysis: Analysis
        do {
            let location = try await session.authorized { api, token in
                try await api.analysisLocation(trackId: trackId, token: token)
            }
            let data = try await client.download(location.url)
            analysis = try analysisStore.save(data, for: trackId)
        } catch let error as APIError where error.status == 404 {
            markFailed(trackId, message: Transcription.analysisFailedMessage)
            return
        } catch is DecodingError {
            // Not a valid v1 analysis file.
            markFailed(trackId, message: Transcription.analysisFailedMessage)
            return
        } catch is AnalysisStore.InvalidTrackID {
            markFailed(trackId, message: Transcription.analysisFailedMessage)
            return
        }
        jobs[trackId] = nil
        guard let item = fetch(trackId) else { return }
        item.apply(analysis)
        save()
    }

    // MARK: - Steps

    private func begin(_ item: LibraryItem, _ job: Job) {
        item.status = .analyzing
        item.jobId = job.jobId
        item.error = nil
        item.updatedAt = .now
        save()
        jobs[job.trackId] = job
        schedulePolling(job.trackId, replace: true)
    }

    private func createJob(source: Analysis.Source, uploadId: String, title: String) async throws -> Job {
        let body = JobRequest(source: source, uploadId: uploadId, title: title)
        return try await session.authorized { api, token in
            try await api.createJob(body, token: token)
        }
    }

    /// SHA-256 → `POST /uploads` → PUT. Returns the uploadId.
    private func upload(_ fileURL: URL) async throws -> String {
        guard let contentType = Transcription.contentType(forExtension: fileURL.pathExtension) else {
            throw TranscriptionError(message: Transcription.unsupportedMessage)
        }
        let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard size > 0 else { throw TranscriptionError(message: Transcription.emptyMessage) }
        let maxBytes = session.config?.maxUploadBytes ?? Transcription.maxUploadBytes
        guard size <= maxBytes else { throw TranscriptionError(message: Transcription.tooLargeMessage) }

        let sha256 = try await Transcription.sha256(of: fileURL)
        let request = UploadRequest(contentType: contentType, sizeBytes: size, sha256: sha256)
        let target = try await session.authorized { api, token in
            try await api.createUpload(request, token: token)
        }
        guard let client = session.api else { throw SessionStore.SessionError.missingBaseURL }
        try await client.uploadFile(fileURL, to: target.putUrl, contentType: contentType)
        return target.uploadId
    }

    private func localAudioURL(_ item: LibraryItem) throws -> URL {
        guard let url = audioURL(for: item), FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            throw TranscriptionError(message: Transcription.missingAudioMessage)
        }
        return url
    }

    /// Renames the pending file to `audio/{trackId}.{ext}` (or drops it if that file already
    /// exists, as it does when the same audio is decoded twice). Returns the file name.
    private func moveToTrack(_ pending: URL, trackId: String) throws -> String {
        guard Transcription.isSafeTrackId(trackId) else {
            throw TranscriptionError(message: Transcription.genericMessage)
        }
        let fileName = "\(trackId).\(pending.pathExtension.lowercased())"
        let target = audioDirectory.appending(path: fileName, directoryHint: .notDirectory)
        let fileManager = FileManager.default
        if pending.standardizedFileURL.path != target.standardizedFileURL.path {
            if fileManager.fileExists(atPath: target.path(percentEncoded: false)) {
                try? fileManager.removeItem(at: pending)
            } else {
                try fileManager.createDirectory(at: audioDirectory, withIntermediateDirectories: true)
                try fileManager.moveItem(at: pending, to: target)
            }
        }
        return fileName
    }

    private func fetch(_ trackId: String) -> LibraryItem? {
        try? context.fetch(LibraryItem.withTrackId(trackId)).first
    }

    private func save() {
        try? context.save()
    }

    /// Rethrows any failure as a `TranscriptionError` with a user-facing message.
    private func userFacing<T: Sendable>(_ body: () async throws -> T) async throws -> T {
        do {
            return try await body()
        } catch let error as CancellationError {
            throw error
        } catch let error as TranscriptionError {
            throw error
        } catch {
            throw TranscriptionError(message: Transcription.userMessage(for: error))
        }
    }
}

#if DEBUG
extension TranscriptionService {
    /// Kept alive for the life of the preview process (a context needs its container).
    private static let previewContainer: ModelContainer = {
        do {
            return try ModelContainer(for: LibraryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        } catch {
            fatalError("Preview container: \(error)")
        }
    }()

    /// For SwiftUI previews: an in-memory library, no API, files in a temporary folder.
    static func preview() -> TranscriptionService {
        let root = FileManager.default.temporaryDirectory.appending(path: "preview", directoryHint: .isDirectory)
        return TranscriptionService(
            session: SessionStore(api: nil, secureStore: PreviewSecureStore()),
            context: previewContainer.mainContext,
            analysisStore: AnalysisStore(directory: root.appending(path: "analysis", directoryHint: .isDirectory)),
            audioDirectory: root.appending(path: "audio", directoryHint: .isDirectory)
        )
    }
}

extension SessionStore {
    /// For SwiftUI previews: no API, nothing stored.
    static func preview() -> SessionStore {
        SessionStore(api: nil, secureStore: PreviewSecureStore())
    }
}

@MainActor
private final class PreviewSecureStore: SecureStore {
    func string(for key: String) -> String? { nil }
    func set(_ value: String?, for key: String) {}
}
#endif
