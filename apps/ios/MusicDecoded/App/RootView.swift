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
    }
}

#Preview {
    RootView()
}
