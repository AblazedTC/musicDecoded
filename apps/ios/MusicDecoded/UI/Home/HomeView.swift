import SwiftUI

/// Placeholder. The real Home screen is T-203i; the source cards below are the entry points it
/// will keep (Files, Media Library, Record; `SourceRoute`).
struct HomeView: View {
    @State private var sourceRoute: SourceRoute?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.lg) {
                    VStack(alignment: .leading, spacing: Spacing.sm) {
                        SectionHeader(title: "Recently Decoded")
                        Text("Songs you decode will show up here.")
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondary)
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
        }
    }
}

#Preview {
    HomeView()
        .environment(TranscriptionService.preview())
        .preferredColorScheme(.dark)
}
