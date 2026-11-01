import Foundation
import Testing
@testable import MusicDecoded

/// The in-app YouTube browser's state (T-308i).
@MainActor
struct YouTubeBrowserTests {
    @Test func noVideoOnTheFeedOrPlaylists() {
        let model = YouTubeBrowserModel()
        #expect(model.videoId == nil)
        model.url = URL(string: "https://m.youtube.com/")
        #expect(model.videoId == nil)
        model.url = URL(string: "https://m.youtube.com/playlist?list=PL1234567890")
        #expect(model.videoId == nil)
        #expect(model.video == nil)
    }

    @Test func videoPageGivesTheVideoWithThePageTitle() {
        let model = YouTubeBrowserModel()
        model.url = URL(string: "https://m.youtube.com/watch?v=dQw4w9WgXcQ&pp=abc")
        model.pageTitle = "(2) Sunny Days - YouTube"

        #expect(model.videoId == "dQw4w9WgXcQ")
        #expect(model.video?.videoId == "dQw4w9WgXcQ")
        #expect(model.video?.title == "Sunny Days")
        #expect(model.video?.thumbnailUrl.absoluteString == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg")
    }

    @Test func transcribeStartsTheJobForTheVideoOnScreen() async {
        let model = YouTubeBrowserModel()
        model.url = URL(string: "https://m.youtube.com/shorts/dQw4w9WgXcQ")
        model.pageTitle = "YouTube"
        var started: [VideoSummary] = []

        let trackId = await model.transcribe { video in
            started.append(video)
            return "track-1"
        }

        #expect(trackId == "track-1")
        #expect(started.map(\.videoId) == ["dQw4w9WgXcQ"])
        // A generic page title isn't a song title; the Library uses its fallback.
        #expect(started.first?.title == "")
        #expect(model.isStarting == false)
        #expect(model.errorMessage == nil)
    }

    @Test func transcribeDoesNothingWithoutAVideo() async {
        let model = YouTubeBrowserModel()
        model.url = URL(string: "https://m.youtube.com/@SomeChannel")
        var calls = 0

        let trackId = await model.transcribe { _ in
            calls += 1
            return "track-1"
        }

        #expect(trackId == nil)
        #expect(calls == 0)
    }

    @Test func transcribeFailureShowsAMessage() async {
        let model = YouTubeBrowserModel()
        model.url = URL(string: "https://youtu.be/dQw4w9WgXcQ")

        let trackId = await model.transcribe { _ in
            throw APIError(code: .rateLimited, message: "", status: 429)
        }

        #expect(trackId == nil)
        #expect(model.errorMessage == Transcription.rateLimitedMessage)
        #expect(model.isStarting == false)

        // Same video, new params: the error stays. Another video: it's cleared.
        model.url = URL(string: "https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=10s")
        #expect(model.errorMessage == Transcription.rateLimitedMessage)
        model.url = URL(string: "https://m.youtube.com/watch?v=aaaaaaaaaaa")
        #expect(model.errorMessage == nil)
    }

    @Test func newURLDropsTheOldTitle() {
        let model = YouTubeBrowserModel()
        model.url = URL(string: "https://m.youtube.com/watch?v=dQw4w9WgXcQ")
        model.pageTitle = "Old Song - YouTube"

        model.url = URL(string: "https://m.youtube.com/watch?v=aaaaaaaaaaa")

        #expect(model.pageTitle == nil)
        #expect(model.video?.title == "")
    }
}
