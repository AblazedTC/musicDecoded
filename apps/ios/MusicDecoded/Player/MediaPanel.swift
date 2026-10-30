import SwiftUI

/// The media panel (DESIGN.md §3.7 "Owner revision 2"): the YouTube video (16:9) for youtube
/// sources, otherwise the artwork centered on ink, letterboxed. The video itself is drawn by
/// `PlayerVideoHost` over the slot this panel marks, so the web view never moves between views.
struct MediaPanel: View {
    let model: PlayerModel

    var body: some View {
        Group {
            if model.videoPlayer is YouTubePlayerController {
                Palette.black
                    .anchorPreference(key: VideoSlotKey.self, value: .bounds) { $0 }
            } else {
                PlayerArtwork(track: model.track, fit: true)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Palette.black)
            }
        }
        .clipShape(.rect)
    }
}

/// The collapsed media panel (less than 120 points left): artwork thumbnail, title, and a button
/// that opens the media as a sheet.
struct MediaStrip: View {
    let model: PlayerModel
    let onOpen: () -> Void

    var body: some View {
        let isVideo = model.videoPlayer is YouTubePlayerController
        HStack(spacing: Spacing.sm) {
            PlayerArtwork(track: model.track, fit: false)
                .frame(width: ComponentSize.mediaStripArt, height: ComponentSize.mediaStripArt)
                .clipShape(.rect)
                .squareBorder(Palette.borderOnInk)
            Text(model.track?.title ?? "")
                .font(.labelSmall)
                .foregroundStyle(Palette.text)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(action: onOpen) {
                Text(isVideo ? "Show video" : "Show artwork")
                    .font(.labelSmall)
                    .foregroundStyle(Palette.onPrimary)
                    .lineLimit(1)
                    .padding(.horizontal, Spacing.sm)
                    .frame(minHeight: ComponentSize.minTouchTarget)
            }
            .buttonStyle(BlockButtonStyle(fill: Palette.primary, shadow: 0))
        }
        .padding(.horizontal, Spacing.md)
        .frame(maxWidth: .infinity, minHeight: ComponentSize.mediaStripHeight, maxHeight: ComponentSize.mediaStripHeight)
        .background(Palette.surface)
    }
}

/// The media as a sheet, from the strip: the video (it moves here while the sheet is open) or
/// the artwork.
struct MediaSheet: View {
    let model: PlayerModel

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let controller = model.videoPlayer as? YouTubePlayerController {
                    YouTubePlayerView(controller: controller)
                        .aspectRatio(16 / 9, contentMode: .fit)
                        .accessibilityLabel("YouTube video")
                } else {
                    PlayerArtwork(track: model.track, fit: true)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Palette.black)
            .navigationTitle(model.track?.title ?? "")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Palette.canvas, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .preferredColorScheme(.dark)
    }
}

/// The track's thumbnail, or its initial-letter art when there's none (or it fails to load).
/// `fit` letterboxes it inside the frame; otherwise it fills.
struct PlayerArtwork: View {
    let track: PlayerTrack?
    let fit: Bool

    var body: some View {
        let title = track?.title ?? ""
        GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            let letterFont = Font.display(max(side / 2, 11), relativeTo: .largeTitle)
            Group {
                if let url = track?.artworkURL {
                    AsyncImage(url: url) { image in
                        image
                            .resizable()
                            .aspectRatio(contentMode: fit ? .fit : .fill)
                    } placeholder: {
                        InitialArt(title: title, size: side, font: letterFont)
                    }
                } else {
                    InitialArt(title: title, size: side, font: letterFont)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .accessibilityHidden(true)
    }
}

/// Where the media panel wants the video, in the coordinates of whoever reads it.
struct VideoSlotKey: PreferenceKey {
    static var defaultValue: Anchor<CGRect>? { nil }

    static func reduce(value: inout Anchor<CGRect>?, nextValue: () -> Anchor<CGRect>?) {
        value = value ?? nextValue()
    }
}

/// The one place the player's YouTube video is drawn: over the media panel's slot when there is
/// one, else mounted at 1 pt and nearly transparent, so a video that drives the clock keeps
/// playing in every view. Always the same view, so switching views never reloads the embed.
struct PlayerVideoHost: View {
    let controller: YouTubePlayerController
    let slot: Anchor<CGRect>?

    var body: some View {
        GeometryReader { proxy in
            let frame = slot.map { Self.fitted(proxy[$0]) } ?? CGRect(x: 0, y: 0, width: 1, height: 1)
            let visible = slot != nil
            YouTubePlayerView(controller: controller)
                .frame(width: frame.width, height: frame.height)
                .opacity(visible ? 1 : 0.01)
                .allowsHitTesting(visible)
                .accessibilityHidden(visible == false)
                .accessibilityLabel("YouTube video")
                .position(x: frame.midX, y: frame.midY)
        }
    }

    /// The largest 16:9 rectangle centered in `slot`.
    static func fitted(_ slot: CGRect) -> CGRect {
        guard slot.width > 0, slot.height > 0 else { return CGRect(x: slot.minX, y: slot.minY, width: 1, height: 1) }
        let width = min(slot.width, slot.height * 16 / 9)
        let height = width * 9 / 16
        return CGRect(x: slot.midX - width / 2, y: slot.midY - height / 2, width: width, height: height)
    }
}
