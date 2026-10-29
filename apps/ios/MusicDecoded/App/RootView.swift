import SwiftUI

/// The bottom tabs (DESIGN.md §2; 3 tabs until the Tuner decision, T-505).
enum AppTab: Hashable {
    case home, library, settings
}

/// The three tabs from PLAN.md §3.1. Each tab owns its `NavigationStack`.
struct RootView: View {
    @State private var tab = AppTab.home

    var body: some View {
        TabView(selection: $tab) {
            HomeView(
                onShowLibrary: { tab = .library },
                onShowSettings: { tab = .settings }
            )
            .tabItem { Label("Home", systemImage: "house") }
            .tag(AppTab.home)

            LibraryView()
                .tabItem { Label("Library", systemImage: "music.note.list") }
                .tag(AppTab.library)

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(AppTab.settings)
        }
        .tint(Palette.primary)
        .preferredColorScheme(.dark) // v1 is dark-only (DESIGN.md §1)
    }
}

#if DEBUG
#Preview {
    RootView()
        .environment(TranscriptionService.preview())
        .environment(SessionStore.preview())
        .environment(SyncService.preview())
        .environment(SettingsStore())
        .environment(NowPlaying())
        .modelContainer(for: LibraryItem.self, inMemory: true)
}
#endif
