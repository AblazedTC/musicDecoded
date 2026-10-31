import MusicDecodedCore
import SwiftUI

/// DESIGN.md §2: the mini player above the tab bar on Home and Library while `NowPlaying` has a
/// track. Art (initial), title, "artist · key", play/pause, dismiss and a 2pt progress line.
/// Tapping it opens the Player.
struct MiniPlayer: View {
    let onOpen: (_ trackId: String) -> Void

    @Environment(NowPlaying.self) private var nowPlaying

    private static let tickInterval = Duration.milliseconds(250)
    private static let progressLineHeight: CGFloat = 2

    var body: some View {
        if let track = nowPlaying.track {
            VStack(spacing: 0) {
                MiniPlayerProgress(durationSec: track.durationSec)
                    .frame(height: Self.progressLineHeight)

                HStack(spacing: Spacing.sm) {
                    Button {
                        onOpen(track.trackId)
                    } label: {
                        HStack(spacing: Spacing.sm) {
                            InitialArt(title: track.title, size: ComponentSize.iconButton, font: .rowTitle)
                            VStack(alignment: .leading, spacing: 0) {
                                Text(track.title)
                                    .font(.rowTitle)
                                    .foregroundStyle(Palette.text)
                                    .lineLimit(1)
                                let subtitle = Self.subtitle(track)
                                if subtitle.isEmpty == false {
                                    Text(subtitle)
                                        .font(.detail)
                                        .foregroundStyle(Palette.textSecondary)
                                        .lineLimit(1)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .contentShape(.rect)
                    }
                    .buttonStyle(.plain)
                    .accessibilityHint("Opens the player")

                    IconButton(
                        title: nowPlaying.isPlaying ? "Pause" : "Play",
                        systemImage: nowPlaying.isPlaying ? "pause.fill" : "play.fill",
                        isActive: true
                    ) {
                        nowPlaying.togglePlay()
                    }
                    IconButton(title: "Close player", systemImage: "xmark", size: .small) {
                        nowPlaying.dismiss()
                    }
                }
                .padding(.leading, Spacing.xs)
                .padding(.vertical, Spacing.xxs)
            }
            .background(Palette.surface, in: .rect)
            .clipShape(.rect)
            .squareBorder(Palette.border)
            .hardShadow(HardShadow.sm, color: Palette.violet)
            .padding(.leading, Spacing.xs)
            .padding(.trailing, Spacing.xs + HardShadow.sm)
            .padding(.top, Spacing.xxs)
            .padding(.bottom, Spacing.xxs + HardShadow.sm)
            // Keeps the progress line and play/pause icon current while the Player isn't on screen.
            .task(id: track.trackId) {
                while Task.isCancelled == false {
                    nowPlaying.tick()
                    do {
                        try await Task.sleep(for: Self.tickInterval)
                    } catch {
                        return
                    }
                }
            }
        }
    }

    /// "artist · key".
    static func subtitle(_ track: NowPlaying.Track) -> String {
        [track.artist, track.key.map(PlayerFormat.key)]
            .compactMap { $0 }
            .joined(separator: " · ")
    }
}

/// The 2pt line on top; reads `position` so only it redraws on each tick.
private struct MiniPlayerProgress: View {
    let durationSec: Double

    @Environment(NowPlaying.self) private var nowPlaying

    var body: some View {
        let fraction = durationSec > 0 ? min(max(nowPlaying.position / durationSec, 0), 1) : 0
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Palette.surfaceRaised
                Palette.primary
                    .frame(width: proxy.size.width * fraction)
            }
        }
        .accessibilityHidden(true)
    }
}

extension View {
    /// Puts the mini player above the tab bar (use on a tab's root content).
    func miniPlayerInset(onOpen: @escaping (_ trackId: String) -> Void) -> some View {
        safeAreaInset(edge: .bottom, spacing: 0) {
            MiniPlayer(onOpen: onOpen)
        }
    }
}
