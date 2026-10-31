import SwiftUI
import WebKit

/// Shows a `YouTubePlayerController`'s web view. The controller owns the view, so dismantling
/// this representable leaves the player (and its state) alone. The web view sits in a container
/// and is moved into it whenever it was last shown elsewhere (the media panel, the media sheet),
/// so only one place ever owns it and a torn-down host can't take it along.
struct YouTubePlayerView: UIViewRepresentable {
    let controller: YouTubePlayerController

    func makeUIView(context: Context) -> UIView {
        let container = UIView()
        container.backgroundColor = .black
        container.clipsToBounds = true
        Self.attach(controller.webView, to: container)
        return container
    }

    func updateUIView(_ container: UIView, context: Context) {
        if controller.webView.superview !== container {
            Self.attach(controller.webView, to: container)
        }
    }

    private static func attach(_ webView: WKWebView, to container: UIView) {
        webView.removeFromSuperview()
        webView.frame = container.bounds
        webView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        container.addSubview(webView)
    }
}

/// The Video tab of the visualizer card (DESIGN.md §3.7 item 6): the 16:9 embed with the
/// "Authorized embedded playback" badge. When hidden it stays mounted at 1 pt, so a video that
/// drives the clock keeps playing under the Piano tab.
struct VideoPanel: View {
    let controller: YouTubePlayerController
    let isVisible: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            YouTubePlayerView(controller: controller)
                .aspectRatio(16 / 9, contentMode: .fit)
                .frame(width: isVisible ? nil : 1, height: isVisible ? nil : 1)
                .clipShape(.rect)
                .opacity(isVisible ? 1 : 0.01)
                .allowsHitTesting(isVisible)
                .accessibilityHidden(isVisible == false)
                .accessibilityLabel("YouTube video")

            if isVisible {
                Label("Authorized embedded playback", systemImage: "checkmark.shield")
                    .font(.overline)
                    .foregroundStyle(Palette.textSecondary)
            }
        }
    }
}
