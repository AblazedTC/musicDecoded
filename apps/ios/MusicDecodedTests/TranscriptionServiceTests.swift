import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

/// The job runner against a stubbed API, an in-memory library and a temporary folder.
@MainActor
@Suite(.serialized)
struct TranscriptionServiceTests {
    private final class BundleToken {}

    let root: URL
    let container: ModelContainer
    let analysisStore: AnalysisStore
    let audioDirectory: URL
    let fixture: Data
    let trackId: String

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appending(path: "TranscriptionServiceTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        container = try ModelContainer(
            for: LibraryItem.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        analysisStore = try AnalysisStore(directory: AppDirectories.analysis(in: root))
        audioDirectory = try AppDirectories.audio(in: root)
        let url = try #require(Bundle(for: BundleToken.self).url(forResource: "sample.analysis", withExtension: "json"))
        fixture = try Data(contentsOf: url)
        trackId = try Analysis(data: fixture).meta.trackId
    }

    // MARK: - Helpers

    private func makeService(_ handler: @escaping StubURLProtocol.Handler) -> TranscriptionService {
        let session = SessionStore(
            api: StubURLProtocol.client(handler),
            secureStore: InMemorySecureStore(["deviceId": "device-1", "token": "token-1"]),
            defaults: UserDefaults(suiteName: "TranscriptionServiceTests-\(UUID().uuidString)")!
        )
        return TranscriptionService(
            session: session,
            context: container.mainContext,
            analysisStore: analysisStore,
            audioDirectory: audioDirectory,
            timing: .init(fast: .milliseconds(1), slow: .milliseconds(1), backOffAfter: .milliseconds(5))
        )
    }

    /// A pending recording with a few bytes in it.
    private func makePendingAudio(_ service: TranscriptionService) throws -> LocalAudio {
        let url = try service.newPendingFileURL(extension: "m4a")
        try Data("not really audio".utf8).write(to: url)
        return LocalAudio(fileURL: url, title: "My take", source: .recording, durationSec: 12)
    }

    private func item(_ trackId: String) throws -> LibraryItem? {
        try container.mainContext.fetch(LibraryItem.withTrackId(trackId)).first
    }

    private func job(_ status: String, error: String? = nil) -> Data {
        let errorJSON = error.map { "\"\($0)\"" } ?? "null"
        return json("""
        {"jobId": "job-1", "trackId": "\(trackId)", "status": "\(status)", "stage": null, "progress": 0, "error": \(errorJSON)}
        """)
    }

    /// Answers the full happy path; `jobStatuses` are returned by successive `GET /jobs/job-1`.
    private func happyPath(
        log: RequestLog,
        jobStatuses: [String],
        createJob: @escaping @Sendable (Int) -> (Int, Data)? = { _ in nil }
    ) -> StubURLProtocol.Handler {
        let jobBodies = jobStatuses.map { job($0) }
        let completeJob = job("complete")
        let fixture = fixture
        let trackId = trackId
        return { request in
            let earlier = log.record(request)
            let host = request.url?.host() ?? ""
            switch (request.httpMethod, request.url?.path()) {
            case ("POST", "/uploads"):
                return (201, json("""
                {"uploadId": "upload-\(earlier + 1)", "putUrl": "https://\(host)/put/upload-\(earlier + 1)", "expiresAt": "2030-01-01T00:00:00Z"}
                """))
            case ("PUT", _):
                return (200, Data())
            case ("POST", "/jobs"):
                if let override = createJob(earlier) { return override }
                return (202, jobBodies.first ?? completeJob)
            case ("GET", "/jobs/job-1"):
                return (200, earlier < jobBodies.count ? jobBodies[earlier] : completeJob)
            case ("GET", "/tracks/\(trackId)/analysis"):
                return (200, json("""
                {"url": "https://\(host)/files/\(trackId).json", "schemaVersion": 1}
                """))
            case ("GET", "/files/\(trackId).json"):
                return (200, fixture)
            default:
                return (404, SampleJSON.notFound)
            }
        }
    }

    // MARK: - Tests

    @Test func completeImmediately() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["complete"]))
        let audio = try makePendingAudio(service)

        let started = try await service.start(audio)
        await service.pollingTask(for: started)?.value

        #expect(started == trackId)
        #expect(log.routes == [
            "POST /uploads", "PUT /put/upload-1", "POST /jobs", "GET /jobs/job-1",
            "GET /tracks/\(trackId)/analysis", "GET /files/\(trackId).json",
        ])

        let uploadBody = try #require(log.all.first?.body)
        let upload = try JSONDecoder().decode([String: AnyCodableValue].self, from: uploadBody)
        #expect(upload["contentType"] == .string("audio/mp4"))
        #expect(upload["sizeBytes"] == .int(16))
        let expectedSha = try await Transcription.sha256(of: audioDirectory.appending(path: "\(trackId).m4a"))
        #expect(upload["sha256"] == .string(expectedSha))
        // The presigned PUT carries no bearer token.
        #expect(log.all[1].authorization == nil)

        let jobBody = try #require(log.all[2].body)
        let jobRequest = try JSONDecoder().decode([String: String].self, from: jobBody)
        #expect(jobRequest == ["source": "recording", "uploadId": "upload-1", "title": "My take"])

        let saved = try #require(try item(trackId))
        let analysis = try Analysis(data: fixture)
        #expect(saved.status == .ready)
        #expect(saved.jobId == nil)
        #expect(saved.error == nil)
        #expect(saved.title == "My take")
        #expect(saved.source == .recording)
        #expect(saved.audioUploadId == "upload-1")
        #expect(saved.durationSec == analysis.meta.durationSec)
        #expect(saved.bpm == analysis.bpm)
        #expect(saved.key == analysis.key.map { "\($0.tonic) \($0.mode.rawValue)" })
        #expect(saved.localAudioPath == "\(trackId).m4a")
        #expect(FileManager.default.fileExists(atPath: audioDirectory.appending(path: "\(trackId).m4a").path(percentEncoded: false)))
        #expect(FileManager.default.fileExists(atPath: audio.fileURL.path(percentEncoded: false)) == false)
        #expect(try analysisStore.load(trackId: trackId) == analysis)
        #expect(service.jobs[trackId] == nil)
    }

    @Test func queuedThenComplete() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["queued", "processing", "processing"]))

        let started = try await service.start(makePendingAudio(service))
        #expect(try item(started)?.status == .analyzing)
        #expect(try item(started)?.jobId == "job-1")
        await service.pollingTask(for: started)?.value

        #expect(log.routes.filter { $0 == "GET /jobs/job-1" }.count == 4)
        #expect(try item(started)?.status == .ready)
    }

    @Test func failedThenRetryReusesTheUpload() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let failedJob = job("failed", error: "No audio detected")
        let base = happyPath(log: RequestLog(), jobStatuses: ["queued"])
        let service = makeService { request in
            let earlier = log.record(request)
            if request.httpMethod == "GET", request.url?.path() == "/jobs/job-1", earlier == 0 {
                return (200, failedJob)
            }
            return try base(request)
        }

        let started = try await service.start(makePendingAudio(service))
        await service.pollingTask(for: started)?.value

        let failed = try #require(try item(started))
        #expect(failed.status == .failed)
        #expect(failed.error == "No audio detected")
        #expect(failed.jobId == nil)

        try await service.retry(trackId: started)
        await service.pollingTask(for: started)?.value

        #expect(log.routes.filter { $0 == "POST /uploads" }.count == 1)
        #expect(log.routes.filter { $0 == "POST /jobs" }.count == 2)
        #expect(try item(started)?.status == .ready)
        #expect(try item(started)?.error == nil)
    }

    @Test func retryUploadsAgainWhenTheUploadIsGone() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let failedJob = job("failed")
        let base = happyPath(log: RequestLog(), jobStatuses: ["queued"])
        let service = makeService { request in
            let earlier = log.record(request)
            switch (request.httpMethod, request.url?.path()) {
            case ("GET", "/jobs/job-1") where earlier == 0:
                return (200, failedJob)
            case ("POST", "/jobs") where earlier == 1:
                return (404, SampleJSON.notFound)
            default:
                return try base(request)
            }
        }

        let started = try await service.start(makePendingAudio(service))
        await service.pollingTask(for: started)?.value
        #expect(try item(started)?.error == Transcription.analysisFailedMessage)

        try await service.retry(trackId: started)
        await service.pollingTask(for: started)?.value

        #expect(log.routes.filter { $0 == "POST /uploads" }.count == 2)
        #expect(log.routes.filter { $0 == "POST /jobs" }.count == 3)
        #expect(try item(started)?.status == .ready)
    }

    @Test func rateLimitedShowsTheDailyLimitMessage() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let rateLimited = json(#"{"code": "rate_limited", "message": "Daily limit reached"}"#)
        let service = makeService(happyPath(log: log, jobStatuses: []) { _ in (429, rateLimited) })
        let audio = try makePendingAudio(service)

        await #expect(throws: TranscriptionError(message: Transcription.rateLimitedMessage)) {
            try await service.start(audio)
        }

        #expect(try container.mainContext.fetchCount(FetchDescriptor<LibraryItem>()) == 0)
        // The pending file stays so the user can try again from the Import sheet.
        #expect(FileManager.default.fileExists(atPath: audio.fileURL.path(percentEncoded: false)))
    }

    @Test func resumePendingPollsAnalyzingItems() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["processing"]))
        container.mainContext.insert(LibraryItem(trackId: trackId, title: "Left running", source: .file, jobId: "job-1"))
        try container.mainContext.save()

        service.resumePending()
        await service.pollingTask(for: trackId)?.value

        #expect(try item(trackId)?.status == .ready)
    }

    @Test func contentTypes() {
        #expect(Transcription.contentType(forExtension: "M4A") == "audio/mp4")
        #expect(Transcription.contentType(forExtension: "mp3") == "audio/mpeg")
        #expect(Transcription.contentType(forExtension: "wav") == "audio/wav")
        #expect(Transcription.contentType(forExtension: "aiff") == nil)
    }
}

/// UTF-8 bytes of a JSON literal. A free function so `@Sendable` stub handlers can call it.
private func json(_ text: String) -> Data {
    Data(text.utf8)
}

/// Just enough JSON to compare a mixed-type object.
private enum AnyCodableValue: Decodable, Equatable {
    case string(String)
    case int(Int)

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) {
            self = .int(int)
        } else {
            self = .string(try container.decode(String.self))
        }
    }
}
