import SwiftUI
import WebKit

/// The official YouTube embed, playing inline (`playsinline=1`). Used by the video detail hero.
struct YouTubeEmbedView: UIViewRepresentable {
    let videoId: String

    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView(frame: .zero, configuration: YouTubeEmbed.configuration())
        YouTubeEmbed.style(webView)
        webView.loadHTMLString(YouTubeEmbed.iframePage(videoId: videoId), baseURL: YouTubeEmbed.origin)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}
}

/// Shared WebView setup and HTML for the embeds.
enum YouTubeEmbed {
    /// The page's origin. YouTube's embed needs one (it rejects pages with no referrer), and the
    /// IFrame API checks `origin` against it.
    static let origin = URL(string: "https://com.musicdecoded.app")!

    @MainActor
    static func configuration() -> WKWebViewConfiguration {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        // Playback is started by our own controls, outside the web page.
        configuration.mediaTypesRequiringUserActionForPlayback = []
        return configuration
    }

    @MainActor
    static func style(_ webView: WKWebView) {
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never
    }

    /// A page holding just the iframe embed, autoplaying.
    static func iframePage(videoId: String) -> String {
        let src = "https://www.youtube.com/embed/\(videoId)?playsinline=1&autoplay=1&rel=0&origin=\(origin.absoluteString)"
        return """
        <!DOCTYPE html>
        <html><head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <style>html,body{margin:0;padding:0;height:100%;background:transparent;overflow:hidden}iframe{position:absolute;inset:0;width:100%;height:100%;border:0}</style>
        </head><body>
        <iframe src="\(src)" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>
        </body></html>
        """
    }
}
