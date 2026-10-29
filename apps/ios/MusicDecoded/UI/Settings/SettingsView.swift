import SwiftUI

/// Placeholder for Settings (chords, accidentals, account).
struct SettingsView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Settings",
                systemImage: "gearshape",
                description: Text("Options will appear here.")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Palette.canvas)
            .navigationTitle("Settings")
        }
    }
}

#Preview {
    SettingsView()
}
