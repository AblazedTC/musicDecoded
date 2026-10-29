import Foundation
import WebKit

/// The official YouTube IFrame Player API in a `WKWebView` (PLAN.md D-4, T-305i). Commands go in
/// with `evaluateJavaScript`; state and `currentTime` come back through a `WKScriptMessageHandler`.
///
/// Owned by the playback engine (and so by `NowPlaying`), not by a view, so the same player
/// survives the Player screen being closed and reopened; `YouTubePlayerView` just shows `webView`.
@MainActor
final class YouTubePlayerController: VideoPlayerControl {
    let videoId: String
    let webView: WKWebView

    private(set) var isReady = false
    private(set) var hasFailed = false
    private(set) var isPlaying = false

    private var clock = YouTubeVideoClock()
    private var muted = false
    private var rate = 1.0
    /// A seek requested before the player was ready.
    private var pendingSeek: Double?
    private let now: () -> Double
    private let bridge: Bridge

    init(videoId: String, now: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }) {
        self.videoId = videoId
        self.now = now
        let configuration = YouTubeEmbed.configuration()
        let bridge = Bridge()
        // The content controller keeps the bridge; the bridge only weakly points back here.
        configuration.userContentController.add(bridge, name: YouTubePlayerPage.messageName)
        webView = WKWebView(frame: .zero, configuration: configuration)
        self.bridge = bridge
        YouTubeEmbed.style(webView)
        webView.navigationDelegate = bridge
        bridge.controller = self
        webView.loadHTMLString(YouTubePlayerPage.html(videoId: videoId), baseURL: YouTubeEmbed.origin)
    }

    var state: YouTubePlayerState { clock.state }

    var currentTime: Double { clock.position(at: now()) }

    func play() {
        isPlaying = true
        run("player.playVideo()")
    }

    func pause() {
        isPlaying = false
        run("player.pauseVideo()")
    }

    func seek(to seconds: Double) {
        let target = seconds.isFinite ? max(seconds, 0) : 0
        clock.seek(to: target, at: now())
        if isReady {
            run("player.seekTo(\(target), true)")
        } else {
            pendingSeek = target
        }
    }

    func setPlaybackRate(_ rate: Double) {
        self.rate = rate
        run("player.setPlaybackRate(\(rate))")
    }

    func setMuted(_ muted: Bool) {
        self.muted = muted
        run(muted ? "player.mute()" : "player.unMute()")
    }

    /// Applies a message from the page. Internal so tests can drive it.
    func handle(_ message: YouTubeBridgeMessage) {
        let t = now()
        switch message {
        case .ready:
            isReady = true
            hasFailed = false
            applyQueuedCommands()
        case .state(let state, let time):
            clock.report(state: state, time: time, at: t)
            switch state {
            case .playing, .buffering:
                isPlaying = true
            case .paused:
                isPlaying = false
            case .ended, .unstarted, .cued:
                // Ended keeps "playing" like AVPlayer with `actionAtItemEnd = .none`, so a loop
                // that ends with the video can wrap; the player model pauses at the end otherwise.
                break
            }
        case .time(let time):
            clock.report(time: time, at: t)
        case .rate(let rate):
            clock.setRate(rate, at: t)
        case .error:
            hasFailed = true
        }
    }

    /// The page couldn't load (no network).
    func markFailed() {
        hasFailed = true
    }

    private func applyQueuedCommands() {
        run(muted ? "player.mute()" : "player.unMute()")
        if rate != 1 {
            run("player.setPlaybackRate(\(rate))")
        }
        if let pendingSeek {
            self.pendingSeek = nil
            run("player.seekTo(\(pendingSeek), true)")
        }
        if isPlaying {
            run("player.playVideo()")
        }
    }

    private func run(_ script: String) {
        guard isReady else { return }
        webView.evaluateJavaScript(script, completionHandler: nil)
    }
}

/// Receives the page's messages and load failures. A separate object because
/// `WKUserContentController` keeps its handlers alive.
@MainActor
private final class Bridge: NSObject, @preconcurrency WKScriptMessageHandler, @preconcurrency WKNavigationDelegate {
    weak var controller: YouTubePlayerController?

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let parsed = YouTubeBridgeMessage(body: message.body) else { return }
        controller?.handle(parsed)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: any Error) {
        controller?.markFailed()
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: any Error) {
        controller?.markFailed()
    }
}
