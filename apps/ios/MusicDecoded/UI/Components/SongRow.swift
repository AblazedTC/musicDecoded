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
                    .lineLimit(1)

                if let artist {
                    Text(artist)
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                        .lineLimit(1)
                }

                statusLine
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)

            if status == .failed, let onRetry {
                Button(action: onRetry) {
                    Text("Retry")
                        .font(.detail.weight(.semibold))
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

    @ViewBuilder
    private var statusLine: some View {
        switch status {
        case .ready:
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
        case .analyzing(let progress):
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(progress.map { "Analyzing · \($0)%" } ?? "Analyzing…")
                    .font(.overline.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(Palette.primary)
                    .lineLimit(1)
                ProgressView(value: Double(progress ?? 0), total: 100)
                    .tint(Palette.primary)
                    .accessibilityHidden(true)
            }
        case .failed:
            Text("Failed")
                .font(.overline.weight(.semibold))
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
            .background(Palette.surfaceSoft)
            .clipShape(.rect(cornerRadius: Radius.art))
            .accessibilityHidden(true)
        } else {
            InitialArt(title: title)
        }
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
