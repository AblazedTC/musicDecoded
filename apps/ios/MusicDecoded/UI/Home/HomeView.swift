import SwiftUI

/// Placeholder. The real Home screen is T-203i.
struct HomeView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Nothing Decoded Yet",
                systemImage: "waveform",
                description: Text("Songs you transcribe will show up here.")
            )
            .navigationTitle("Home")
        }
    }
}

#Preview {
    HomeView()
}
