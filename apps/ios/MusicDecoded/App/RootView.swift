import SwiftUI

/// The three tabs from PLAN.md §3.1. The mini player and real screens come in later tasks.
struct RootView: View {
    var body: some View {
        TabView {
            HomeView()
                .tabItem { Label("Home", systemImage: "house") }

            LibraryView()
                .tabItem { Label("Library", systemImage: "music.note.list") }

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .preferredColorScheme(.dark) // v1 is dark-only (DESIGN.md §1)
    }
}

#Preview {
    RootView()
}
