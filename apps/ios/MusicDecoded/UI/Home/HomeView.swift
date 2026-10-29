import SwiftData
import SwiftUI

/// Placeholder. The real Home screen is T-203i; the source cards below are the entry points it
/// will keep (Files, Media Library, Record; `SourceRoute`). Ready rows open the Player.
struct HomeView: View {
    @State private var sourceRoute: SourceRoute?
    @Query(LibraryItem.newestFirst) private var items: [LibraryItem]

    /// Up to 3 decoded songs, newest first.
    private var recentReady: [LibraryItem] {
        Array(items.lazy.filter { $0.status == .ready }.prefix(3))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.lg) {
                    VStack(alignment: .leading, spacing: Spacing.sm) {
                        SectionHeader(title: "Recently Decoded")
                        if recentReady.isEmpty {
                            Text("Songs you decode will show up here.")
                                .font(.detail)
                                .foregroundStyle(Palette.textSecondary)
                        } else {
                            ForEach(recentReady) { item in
                                NavigationLink(value: PlayerRoute(trackId: item.trackId)) {
                                    SongRow(
                                        title: item.title,
                                        artist: item.artist,
                                        metadata: SongMetadata.line(durationSec: item.durationSec, key: item.key, bpm: item.bpm),
                                        artworkURL: item.thumbnailUrl.flatMap(URL.init(string:)),
                                        isOffline: item.localAudioPath != nil
                                    )
                                }
                                .buttonStyle(.plain)
                            }
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

                    VStack(alignment: .leading, spacing: Spacing.sm) {
                        SectionHeader(title: "Transcribe Music")
                        HStack(alignment: .top, spacing: Spacing.xs) {
                            SourceCard(title: "Files", subtitle: "Import audio from this device", systemImage: "folder", tint: .files) {
                                sourceRoute = .files
                            }
                            SourceCard(title: "Media Library", subtitle: "Choose local music", systemImage: "music.note.list", tint: .mediaLibrary) {
                                sourceRoute = .mediaLibrary
                            }
                        }
                        SourceCard(title: "Record Audio", subtitle: "Capture a live performance", systemImage: "mic", tint: .record) {
                            sourceRoute = .record
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.lg)
            }
            .background(Palette.canvas)
            .navigationTitle("Home")
            .sourceRoutes($sourceRoute)
            .navigationDestination(for: PlayerRoute.self) { route in
                PlayerScreen(trackId: route.trackId)
            }
        }
    }
}

#Preview {
    HomeView()
        .environment(TranscriptionService.preview())
        .modelContainer(for: LibraryItem.self, inMemory: true)
        .preferredColorScheme(.dark)
}
