import SwiftData
import SwiftUI

/// DESIGN.md §3.1: brand header, Recently Decoded (3 newest + See all), the source grid, the
/// Record card and a rotating tip. Ready rows open the Player; analyzing rows open Processing.
struct HomeView: View {
    let onShowLibrary: () -> Void
    let onShowSettings: () -> Void

    @Environment(SessionStore.self) private var session
    @State private var path = NavigationPath()
    @State private var sourceRoute: SourceRoute?
    /// The 3 newest songs, any status.
    @Query(LibraryItem.recent(limit: 3)) private var recent: [LibraryItem]

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.lg) {
                    BrandHeader(onSearch: onShowLibrary, onProfile: onShowSettings)
                    recentSection
                    transcribeSection
                    TipCard()
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.lg)
            }
            .background(Palette.canvas)
            .toolbar(.hidden, for: .navigationBar)
            .sourceRoutes($sourceRoute, onDecodeStarted: { trackId in
                openProcessing(trackId)
            })
            .navigationDestination(for: PlayerRoute.self) { route in
                PlayerScreen(trackId: route.trackId)
            }
        }
    }

    private var recentSection: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            SectionHeader(title: "Recently Decoded", actionTitle: "See all", action: onShowLibrary)
            if recent.isEmpty {
                Text("Nothing decoded yet. Pick a source below to decode your first song.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
            ForEach(recent) { item in
                LibrarySongRow(
                    item: item,
                    onOpenPlayer: { path.append(PlayerRoute(trackId: $0)) },
                    onOpenProcessing: { openProcessing($0) }
                )
            }
            #if DEBUG
            NavigationLink(value: PlayerRoute(trackId: DebugFixtures.sunnyDays)) {
                Label("Open Sunny Days fixture (Debug)", systemImage: "ladybug")
                    .font(.detail.weight(.semibold))
                    .foregroundStyle(Palette.secondary)
                    .frame(minHeight: ComponentSize.minTouchTarget)
            }
            #endif
        }
    }

    private var transcribeSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            SectionHeader(title: "Transcribe Music")
            Grid(horizontalSpacing: Spacing.sm, verticalSpacing: Spacing.sm) {
                GridRow {
                    SourceCard(title: "Files", subtitle: "Import audio from this device", systemImage: "folder", tint: .files) {
                        sourceRoute = .files
                    }
                    SourceCard(title: "Media Library", subtitle: "Choose local music", systemImage: "music.note.list", tint: .mediaLibrary) {
                        sourceRoute = .mediaLibrary
                    }
                }
                GridRow {
                    // YouTube arrives in M3: shown, but disabled.
                    comingSoon(SourceCard(title: "YouTube", subtitle: "Coming soon", systemImage: "play.rectangle", tint: .youtube) {})
                    if session.config?.soundcloudEnabled == true {
                        comingSoon(SourceCard(title: "SoundCloud", subtitle: "Coming soon", systemImage: "cloud", tint: .soundcloud) {})
                    } else {
                        Color.clear
                            .gridCellUnsizedAxes(.vertical)
                    }
                }
            }
            RecordCard {
                sourceRoute = .record
            }
        }
    }

    private func comingSoon(_ card: SourceCard) -> some View {
        card
            .disabled(true)
            .opacity(0.5)
    }

    private func openProcessing(_ trackId: String) {
        // The Processing screen arrives with T-211i.
        _ = trackId
    }
}

#if DEBUG
#Preview {
    HomeView(onShowLibrary: {}, onShowSettings: {})
        .environment(TranscriptionService.preview())
        .environment(SessionStore.preview())
        .modelContainer(for: LibraryItem.self, inMemory: true)
        .preferredColorScheme(.dark)
}
#endif
