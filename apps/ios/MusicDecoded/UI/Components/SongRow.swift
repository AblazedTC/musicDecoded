import SwiftUI

/// One song in Home's "Recently Decoded" and in the Library: artwork, title, artist,
/// the `duration · key · BPM` line, an offline badge and an optional ⋯ menu.
struct SongRow<Actions: View>: View {
    let title: String
    let artist: String?
    /// Usually built with `SongMetadata.line(durationSec:key:bpm:)`.
    let metadata: String
    let artworkURL: URL?
    let isOffline: Bool
    private let hasActions: Bool
    private let actions: Actions

    /// A row with a ⋯ menu holding `actions`.
    init(
        title: String,
        artist: String?,
        metadata: String,
        artworkURL: URL? = nil,
        isOffline: Bool = false,
        @ViewBuilder actions: () -> Actions
    ) {
        self.init(
            title: title,
            artist: artist,
            metadata: metadata,
            artworkURL: artworkURL,
            isOffline: isOffline,
            hasActions: true,
            actions: actions()
        )
    }

    private init(
        title: String,
        artist: String?,
        metadata: String,
        artworkURL: URL?,
        isOffline: Bool,
        hasActions: Bool,
        actions: Actions
    ) {
        self.title = title
        self.artist = artist
        self.metadata = metadata
        self.artworkURL = artworkURL
        self.isOffline = isOffline
        self.hasActions = hasActions
        self.actions = actions
    }

    var body: some View {
        HStack(spacing: Spacing.sm) {
            artwork

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(title)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.text)
                    .lineLimit(1)

                if let artist {
                    Text(artist)
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                        .lineLimit(1)
                }

                HStack(spacing: Spacing.xxs) {
                    if isOffline {
                        offlineBadge
                    }
                    Text(metadata)
                        .font(.overline)
                        .monospacedDigit()
                        .foregroundStyle(Palette.textMuted)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)

            if hasActions {
                Menu {
                    actions
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(Palette.textSecondary)
                        .frame(width: ComponentSize.minTouchTarget, height: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .accessibilityLabel("More actions")
            }
        }
        .padding(Spacing.xs)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.row))
    }

    private var artwork: some View {
        AsyncImage(url: artworkURL) { image in
            image
                .resizable()
                .scaledToFill()
        } placeholder: {
            Image(systemName: "music.note")
                .font(.title3)
                .foregroundStyle(Palette.textMuted)
        }
        .frame(width: ComponentSize.songArt, height: ComponentSize.songArt)
        .background(Palette.surfaceSoft)
        .clipShape(.rect(cornerRadius: Radius.art))
        .accessibilityHidden(true)
    }

    private var offlineBadge: some View {
        Image(systemName: "arrow.down")
            .font(.caption2.weight(.bold))
            .foregroundStyle(Palette.primary)
            .padding(Spacing.xxs)
            .background(Palette.primarySoft, in: .circle)
            .accessibilityLabel("Available offline")
    }
}

extension SongRow where Actions == EmptyView {
    /// A row without a ⋯ menu.
    init(title: String, artist: String?, metadata: String, artworkURL: URL? = nil, isOffline: Bool = false) {
        self.init(
            title: title,
            artist: artist,
            metadata: metadata,
            artworkURL: artworkURL,
            isOffline: isOffline,
            hasActions: false,
            actions: EmptyView()
        )
    }
}

#Preview {
    // Values from contracts/fixtures/sample.analysis.json (duration 263.407s, D major, 136.1 BPM).
    VStack(spacing: Spacing.xs) {
        SongRow(
            title: "Sample track",
            artist: "Unknown artist",
            metadata: SongMetadata.line(durationSec: 263.407, key: "D major", bpm: 136.1),
            isOffline: true
        ) {
            Button("Favorite", systemImage: "heart") {}
            Button("Re-analyze", systemImage: "arrow.clockwise") {}
            Button("Delete", systemImage: "trash", role: .destructive) {}
        }
        SongRow(title: "Voice memo", artist: nil, metadata: SongMetadata.line(durationSec: 42, key: nil, bpm: nil))
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
