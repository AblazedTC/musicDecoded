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

    /// A job that reports track metadata (contracts/openapi.yaml `Job.title/thumbnailUrl/durationSec`).
    private func job(_ status: String, title: String?, thumbnailUrl: String?) -> Data {
        let titleJSON = title.map { "\"\($0)\"" } ?? "null"
        let thumbJSON = thumbnailUrl.map { "\"\($0)\"" } ?? "null"
        return json("""
        {"jobId": "job-1", "trackId": "\(trackId)", "status": "\(status)", "stage": null, "progress": 0, "error": null, "title": \(titleJSON), "thumbnailUrl": \(thumbJSON), "durationSec": 213.0}
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

    @Test func youtubeJobSkipsTheUpload() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["queued"]))
        let video = VideoSummary(
            videoId: "dQw4w9WgXcQ",
            title: "Never Gonna Give You Up",
            channelTitle: "Rick Astley",
            thumbnailUrl: URL(string: "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg")!,
            durationSec: 213
        )

        let started = try await service.startYouTube(video)
        #expect(started == trackId)
        #expect(try item(started)?.status == .analyzing)
        await service.pollingTask(for: started)?.value

        #expect(log.routes.contains("POST /uploads") == false)
        #expect(log.routes.first == "POST /jobs")
        let jobBody = try #require(log.all.first?.body)
        let jobRequest = try JSONDecoder().decode([String: String].self, from: jobBody)
        #expect(jobRequest == ["source": "youtube", "videoId": "dQw4w9WgXcQ"])

        let saved = try #require(try item(trackId))
        #expect(saved.status == .ready)
        #expect(saved.source == .youtube)
        #expect(saved.sourceRef == "dQw4w9WgXcQ")
        #expect(saved.title == "Never Gonna Give You Up")
        #expect(saved.artist == "Rick Astley")
        #expect(saved.localAudioPath == nil)
        #expect(saved.audioUploadId == nil)
    }

    @Test func youtubeRetryAsksAgainByVideoId() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["complete"]))
        let failed = LibraryItem(trackId: trackId, title: "Song", source: .youtube, sourceRef: "dQw4w9WgXcQ", status: .failed)
        container.mainContext.insert(failed)

        try await service.retry(trackId: trackId)
        await service.pollingTask(for: trackId)?.value

        let jobBody = try #require(log.all.first?.body)
        let jobRequest = try JSONDecoder().decode([String: String].self, from: jobBody)
        #expect(jobRequest == ["source": "youtube", "videoId": "dQw4w9WgXcQ"])
        #expect(log.routes.contains("POST /uploads") == false)
        #expect(try item(trackId)?.status == .ready)
    }

    @Test func youtubeDailyLimitShowsThePlanMessage() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let service = makeService { _ in
            (429, json(#"{"code": "rate_limited", "message": "Daily limit reached"}"#))
        }
        let video = VideoSummary(pastedVideoId: "dQw4w9WgXcQ")
        await #expect(throws: TranscriptionError(message: Transcription.rateLimitedMessage)) {
            try await service.startYouTube(video)
        }
    }

    @Test func soundcloudJobSendsTheNormalizedURL() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["queued"]))

        let started = try await service.startSoundCloud(
            url: "Listen on #SoundCloud https://m.soundcloud.com/artist-name/my-song?si=abc"
        )
        #expect(started == trackId)
        await service.pollingTask(for: started)?.value

        #expect(log.routes.contains("POST /uploads") == false)
        let jobBody = try #require(log.all.first?.body)
        let jobRequest = try JSONDecoder().decode([String: String].self, from: jobBody)
        #expect(jobRequest == ["source": "soundcloud", "soundcloudUrl": "https://soundcloud.com/artist-name/my-song"])

        let saved = try #require(try item(trackId))
        #expect(saved.status == .ready)
        #expect(saved.source == .soundcloud)
        #expect(saved.sourceRef == "https://soundcloud.com/artist-name/my-song")
        #expect(saved.title == "My Song")
        #expect(saved.artist == "Artist Name")
        #expect(saved.localAudioPath == nil)
    }

    @Test func soundcloudFeatureDisabledShowsAFriendlyMessage() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let service = makeService { _ in
            (403, json(#"{"code": "feature_disabled", "message": "SoundCloud is not enabled"}"#))
        }
        await #expect(throws: TranscriptionError(message: Transcription.soundcloudDisabledMessage)) {
            try await service.startSoundCloud(url: "https://soundcloud.com/artist/song")
        }
        #expect(try item(trackId) == nil)
    }

    @Test func soundcloudRejectsOtherLinksWithoutCallingTheAPI() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["complete"]))
        await #expect(throws: TranscriptionError(message: Transcription.soundcloudInvalidMessage)) {
            try await service.startSoundCloud(url: "https://soundcloud.com/artist/sets/playlist")
        }
        #expect(log.routes.isEmpty)
    }

    @Test func soundcloudRetryAsksAgainByURL() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let service = makeService(happyPath(log: log, jobStatuses: ["complete"]))
        let url = "https://soundcloud.com/artist/song"
        let failed = LibraryItem(trackId: trackId, title: "Song", source: .soundcloud, sourceRef: url, status: .failed)
        container.mainContext.insert(failed)

        try await service.retry(trackId: trackId)
        await service.pollingTask(for: trackId)?.value

        let jobBody = try #require(log.all.first?.body)
        let jobRequest = try JSONDecoder().decode([String: String].self, from: jobBody)
        #expect(jobRequest == ["source": "soundcloud", "soundcloudUrl": url])
        #expect(log.routes.contains("POST /uploads") == false)
        #expect(try item(trackId)?.status == .ready)
    }

    @Test func polledJobMetadataReplacesAPastedLinksPlaceholderTitle() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let queued = job("queued")
        let base = happyPath(log: log, jobStatuses: [], createJob: { _ in (202, queued) })
        let complete = job("complete", title: "Never Gonna Give You Up", thumbnailUrl: "https://i.ytimg.com/vi/x/hq.jpg")
        let service = makeService { request in
            let answer = try base(request)
            if request.httpMethod == "GET", request.url?.path() == "/jobs/job-1" { return (200, complete) }
            return answer
        }

        let video = VideoSummary(pastedVideoId: "dQw4w9WgXcQ")
        let started = try await service.startYouTube(video)
        #expect(try item(started)?.title == Transcription.youtubeFallbackTitle)
        await service.pollingTask(for: started)?.value

        let saved = try #require(try item(started))
        #expect(saved.title == "Never Gonna Give You Up")
        #expect(saved.status == .ready)
    }

    @Test func jobMetadataNeverOverwritesARealTitleOrThumbnail() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let meta = job("complete", title: "Server title", thumbnailUrl: "https://other/thumb.jpg")
        let service = makeService(happyPath(log: log, jobStatuses: [], createJob: { _ in (202, meta) }))
        let video = VideoSummary(
            videoId: "dQw4w9WgXcQ",
            title: "Sunny Days",
            channelTitle: "",
            thumbnailUrl: URL(string: "https://i.ytimg.com/mine.jpg")!,
            durationSec: nil
        )

        let started = try await service.startYouTube(video)
        await service.pollingTask(for: started)?.value

        let saved = try #require(try item(started))
        #expect(saved.title == "Sunny Days")
        #expect(saved.thumbnailUrl == "https://i.ytimg.com/mine.jpg")
    }

    @Test func createdJobMetadataReplacesASoundCloudSlugTitle() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let log = RequestLog()
        let meta = job("queued", title: "Sunny Days (Live)", thumbnailUrl: "https://i1.sndcdn.com/art.jpg")
        let service = makeService(happyPath(log: log, jobStatuses: [], createJob: { _ in (202, meta) }))

        let started = try await service.startSoundCloud(url: "https://soundcloud.com/artist/sunny-days")
        let saved = try #require(try item(started))
        #expect(saved.title == "Sunny Days (Live)")
        #expect(saved.thumbnailUrl == "https://i1.sndcdn.com/art.jpg")
        await service.pollingTask(for: started)?.value
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
