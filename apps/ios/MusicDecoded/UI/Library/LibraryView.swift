import SwiftUI

/// Placeholder for the full history list.
struct LibraryView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Your Library Is Empty",
                systemImage: "music.note.list",
                description: Text("Everything you decode is saved here.")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Palette.canvas)
            .navigationTitle("Library")
        }
    }
}

#Preview {
    LibraryView()
}
