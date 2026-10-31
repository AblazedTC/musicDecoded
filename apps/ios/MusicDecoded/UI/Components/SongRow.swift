import SwiftUI

/// Where a row's analysis stands. Ready rows show the meta line; the others replace it.
enum SongRowStatus: Sendable, Equatable {
    case ready
    /// `progress` is 0–100 once the job has reported it.
    case analyzing(progress: Int?)
    case failed
}

/// One song in Home's "Recently Decoded" and in the Library: artwork, title, artist,
/// the `duration · key · BPM` line (or the analysis status), an offline badge and an optional ⋯ menu.
struct SongRow<Actions: View>: View {
    let title: String
    let artist: String?
    /// Usually built with `SongMetadata.line(durationSec:key:bpm:)`.
    let metadata: String
    let artworkURL: URL?
    let isOffline: Bool
    let status: SongRowStatus
    /// Shown as "Retry" on a failed row.
    let onRetry: (() -> Void)?
    private let hasActions: Bool
    private let actions: Actions
    /// A big faint number behind the row ("01"), set with `numeral(_:)`. Decorative.
    private var numeralText: String?

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    /// Text wraps at accessibility sizes instead of truncating.
    private var lines: Int { dynamicTypeSize.isAccessibilitySize ? 3 : 1 }

    /// A row with a ⋯ menu holding `actions`.
    init(
        title: String,
        artist: String?,
        metadata: String,
        artworkURL: URL? = nil,
        isOffline: Bool = false,
        status: SongRowStatus = .ready,
        onRetry: (() -> Void)? = nil,
        @ViewBuilder actions: () -> Actions
    ) {
        self.init(
            title: title,
            artist: artist,
            metadata: metadata,
            artworkURL: artworkURL,
            isOffline: isOffline,
            status: status,
            onRetry: onRetry,
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
        status: SongRowStatus,
        onRetry: (() -> Void)?,
        hasActions: Bool,
        actions: Actions
    ) {
        self.title = title
        self.artist = artist
        self.metadata = metadata
        self.artworkURL = artworkURL
        self.isOffline = isOffline
        self.status = status
        self.onRetry = onRetry
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
                    .lineLimit(lines)

                if let artist {
                    Text(artist)
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                        .lineLimit(lines)
                }

                statusLine
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)

            if status == .failed, let onRetry {
                Button(action: onRetry) {
                    Text("Retry")
                        .font(.labelSmall)
                        .foregroundStyle(Palette.primary)
                        .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Retry analysis")
            }

            if hasActions {
                Menu {
                    actions
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.label)
                        .foregroundStyle(Palette.textSecondary)
                        .frame(width: ComponentSize.minTouchTarget, height: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .accessibilityLabel("More actions")
            }
        }
        .padding(Spacing.xs)
        .background(alignment: .trailing) {
            if let numeralText {
                BigNumeral(text: numeralText, size: 64)
                    .padding(.trailing, Spacing.xxxl)
            }
        }
        .clipped()
        .background(Palette.surface, in: .rect)
    }

    /// Draws `text` ("01") big and faint behind the row.
    func numeral(_ text: String?) -> Self {
        var copy = self
        copy.numeralText = text
        return copy
    }

    @ViewBuilder
    private var statusLine: some View {
        switch status {
        case .ready:
            HStack(spacing: Spacing.xxs) {
                if isOffline {
                    offlineBadge
                }
                Text(metadata)
                    .font(.detail)
                    .monospacedDigit()
                    .foregroundStyle(Palette.textMuted)
                    .lineLimit(lines)
            }
        case .analyzing(let progress):
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(progress.map { "Analyzing · \($0)%" } ?? "Analyzing…")
                    .font(.overline)
                    .monospacedDigit()
                    .foregroundStyle(Palette.primary)
                    .lineLimit(lines)
                WrappedProgressBar(fraction: Double(progress ?? 0) / 100)
            }
        case .failed:
            Text("Failed")
                .font(.overline)
                .foregroundStyle(Palette.record)
        }
    }

    /// The thumbnail, or the title's initial (as in the mini player) when there's none or it
    /// hasn't loaded.
    @ViewBuilder
    private var artwork: some View {
        if let artworkURL {
            AsyncImage(url: artworkURL) { image in
                image
                    .resizable()
                    .scaledToFill()
            } placeholder: {
                InitialArt(title: title)
            }
            .frame(width: ComponentSize.songArt, height: ComponentSize.songArt)
            .background(Palette.surfaceRaised)
            .clipShape(.rect)
            .squareBorder(Palette.borderOnInk)
            .accessibilityHidden(true)
        } else {
            InitialArt(title: title)
        }
    }

    private var offlineBadge: some View {
        Image(systemName: "arrow.down")
            .font(.caption2.weight(.bold))
            .foregroundStyle(Palette.onColor)
            .padding(Spacing.xxs / 2)
            .block(Palette.yellow, shadow: HardShadow.sm)
            .tilt(Tilt.sticker)
            .accessibilityLabel("Available offline")
    }
}

extension SongRow where Actions == EmptyView {
    /// A row without a ⋯ menu.
    init(
        title: String,
        artist: String?,
        metadata: String,
        artworkURL: URL? = nil,
        isOffline: Bool = false,
        status: SongRowStatus = .ready,
        onRetry: (() -> Void)? = nil
    ) {
        self.init(
            title: title,
            artist: artist,
            metadata: metadata,
            artworkURL: artworkURL,
            isOffline: isOffline,
            status: status,
            onRetry: onRetry,
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
        SongRow(title: "Sample track", artist: "Unknown artist", metadata: "", status: .analyzing(progress: 42))
        SongRow(title: "Voice memo", artist: nil, metadata: "", status: .failed, onRetry: {})
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
