import MusicDecodedCore
import SwiftUI
import UIKit
import WebKit

/// The in-app YouTube browser (T-308i): m.youtube.com in a web view, web back / forward / reload
/// at the bottom, and "Transcribe this" while a single video is open. Nothing is downloaded on
/// the device (PLAN.md R-1): the worker fetches the audio by videoId.
struct YouTubeBrowserScreen: View {
    /// Gets the trackId once the job exists (the caller pushes Processing).
    let onDecodeStarted: (String) -> Void

    @Environment(TranscriptionService.self) private var transcription
    @State private var model = YouTubeBrowserModel()

    var body: some View {
        YouTubeWebView(model: model)
            .background(Palette.canvas)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 0) {
                    if model.videoId != nil {
                        transcribePanel
                    }
                    bottomBar
                }
                .background(Palette.canvas)
            }
            .navigationTitle("YouTube")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Palette.canvas, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar(.hidden, for: .tabBar)
            .onDisappear {
                model.pause()
            }
    }

    private var transcribePanel: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            if let errorMessage = model.errorMessage {
                Text(errorMessage)
                    .font(.detail)
                    .foregroundStyle(Palette.record)
                    .fixedSize(horizontal: false, vertical: true)
            }
            PrimaryButton(title: model.isStarting ? "Starting…" : "Transcribe this", systemImage: "waveform") {
                transcribe()
            }
            .disabled(model.isStarting)
        }
        .padding(.horizontal, Spacing.md)
        .padding(.top, Spacing.sm)
    }

    private var bottomBar: some View {
        HStack(spacing: Spacing.md) {
            IconButton(title: "Back", systemImage: "chevron.left") { model.goBack() }
                .disabled(model.canGoBack == false)
            IconButton(title: "Forward", systemImage: "chevron.right") { model.goForward() }
                .disabled(model.canGoForward == false)
            Spacer(minLength: 0)
            IconButton(title: "Reload", systemImage: "arrow.clockwise") { model.reload() }
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.xs)
    }

    private func transcribe() {
        Task {
            if let trackId = await model.transcribe(using: { try await transcription.startYouTube($0) }) {
                onDecodeStarted(trackId)
            }
        }
    }
}

/// m.youtube.com in a `WKWebView`, reporting its state into `model`.
private struct YouTubeWebView: UIViewRepresentable {
    let model: YouTubeBrowserModel

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = true
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.navigationDelegate = context.coordinator
        model.attach(webView)
        webView.load(URLRequest(url: YouTubeBrowserModel.homeURL))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    func makeCoordinator() -> NavigationPolicy {
        NavigationPolicy()
    }

    /// Keeps the browser on YouTube (and Google sign-in): a tapped link to any other site opens
    /// outside the app, and non-web schemes (app links) are dropped. Embedded frames are left alone.
    @MainActor
    final class NavigationPolicy: NSObject, WKNavigationDelegate {
        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction
        ) async -> WKNavigationActionPolicy {
            if let frame = navigationAction.targetFrame, frame.isMainFrame == false {
                return .allow
            }
            guard let url = navigationAction.request.url else { return .cancel }
            let scheme = url.scheme?.lowercased()
            guard scheme == "https" || scheme == "http" else { return .cancel }
            if YouTubeURL.isAllowedInBrowser(host: url.host) {
                return .allow
            }
            if navigationAction.navigationType == .linkActivated || navigationAction.targetFrame == nil {
                _ = await UIApplication.shared.open(url)
            }
            return .cancel
        }
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        YouTubeBrowserScreen(onDecodeStarted: { _ in })
    }
    .environment(TranscriptionService.preview())
    .preferredColorScheme(.dark)
}
#endif
