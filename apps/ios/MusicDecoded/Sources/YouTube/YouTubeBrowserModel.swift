import Foundation
import MusicDecodedCore
import Observation
import WebKit

/// State for the in-app YouTube browser (T-308i): what page the web view shows, its back/forward
/// state, and starting a `youtube` job for the video on screen. The web view reports into it
/// through `attach(_:)`; tests set the page state directly.
@MainActor
@Observable
final class YouTubeBrowserModel {
    static let homeURL = URL(string: "https://m.youtube.com")!

    typealias Start = @MainActor (VideoSummary) async throws -> String

    /// The page's URL. YouTube is a single-page app, so this changes without page loads.
    /// A new URL drops the old page's title (until the title observer reports the new one) and,
    /// for another video, the old error.
    var url: URL? {
        didSet {
            guard url != oldValue else { return }
            pageTitle = nil
            if YouTubeURL.videoId(from: oldValue) != videoId {
                errorMessage = nil
            }
        }
    }
    /// The page's `document.title`.
    var pageTitle: String?
    var canGoBack = false
    var canGoForward = false
    private(set) var isStarting = false
    private(set) var errorMessage: String?

    @ObservationIgnored private weak var webView: WKWebView?
    @ObservationIgnored private var observations: [NSKeyValueObservation] = []

    /// The video on screen, or `nil` for the feed, search, playlists and channels.
    var videoId: String? {
        YouTubeURL.videoId(from: url)
    }

    /// The video to transcribe: its id and the page's title (no channel or duration on the device).
    var video: VideoSummary? {
        guard let videoId else { return nil }
        var video = VideoSummary(pastedVideoId: videoId)
        video.title = YouTubeURL.videoTitle(fromPageTitle: pageTitle) ?? ""
        return video
    }

    // MARK: - Web view

    /// Follows `webView`'s URL, title and history (key-value observing, since the SPA navigates
    /// without loads).
    func attach(_ webView: WKWebView) {
        self.webView = webView
        observations = [
            webView.observe(\.url, options: [.initial, .new]) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.syncURL() }
            },
            webView.observe(\.title, options: [.initial, .new]) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.syncTitle() }
            },
            webView.observe(\.canGoBack, options: [.initial, .new]) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.syncHistory() }
            },
            webView.observe(\.canGoForward, options: [.initial, .new]) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.syncHistory() }
            },
        ]
    }

    private func syncURL() {
        guard let webView else { return }
        url = webView.url
    }

    private func syncTitle() {
        guard let webView else { return }
        pageTitle = webView.title
    }

    private func syncHistory() {
        guard let webView else { return }
        canGoBack = webView.canGoBack
        canGoForward = webView.canGoForward
    }

    func goBack() {
        webView?.goBack()
    }

    func goForward() {
        webView?.goForward()
    }

    func reload() {
        webView?.reload()
    }

    /// Leaving the screen: stop the video (it would keep playing off screen).
    func pause() {
        webView?.pauseAllMediaPlayback(completionHandler: nil)
    }

    // MARK: - Transcribe

    /// "Transcribe this": starts the `youtube` job for the video on screen. Returns the trackId
    /// (the caller opens Processing), or `nil` with `errorMessage` set.
    func transcribe(using start: Start) async -> String? {
        guard let video, isStarting == false else { return nil }
        isStarting = true
        errorMessage = nil
        defer { isStarting = false }
        do {
            return try await start(video)
        } catch {
            // A 429 here is the user's daily plan limit, so its message includes the upsell.
            errorMessage = Transcription.userMessage(for: error)
            return nil
        }
    }
}
