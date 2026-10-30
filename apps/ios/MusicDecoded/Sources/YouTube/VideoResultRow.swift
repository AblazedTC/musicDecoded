import SwiftUI

/// A search result: 132×78 thumbnail with a duration badge, title (2 lines) and channel.
struct VideoResultRow: View {
    let video: VideoSummary

    var body: some View {
        HStack(alignment: .top, spacing: Spacing.sm) {
            VideoThumbnail(url: video.thumbnailUrl)
                .frame(width: ComponentSize.videoThumbnailWidth, height: ComponentSize.videoThumbnailHeight)
                .clipShape(.rect)
                .squareBorder(Palette.borderOnInk)
                .overlay(alignment: .bottomTrailing) {
                    if let duration = video.durationText {
                        DurationBadge(text: duration)
                            .padding(Spacing.xxs)
                    }
                }

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(video.displayTitle)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.text)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                if video.channelTitle.isEmpty == false {
                    Text(video.channelTitle)
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
    }
}

/// A remote thumbnail with a neutral placeholder while it loads (or if it fails).
struct VideoThumbnail: View {
    let url: URL

    var body: some View {
        AsyncImage(url: url) { phase in
            if let image = phase.image {
                image
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    Palette.surfaceRaised
                    Image(systemName: "play.rectangle")
                        .foregroundStyle(Palette.textMuted)
                }
            }
        }
        .accessibilityHidden(true)
    }
}

/// `3:45` on a dark pill over the thumbnail's corner.
struct DurationBadge: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.overline)
            .monospacedDigit()
            .foregroundStyle(Palette.onColor)
            .padding(.horizontal, Spacing.xxs)
            .background(Palette.yellow, in: .rect)
            .accessibilityLabel("Duration \(text)")
    }
}
