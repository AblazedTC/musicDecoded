import SwiftUI

/// Placeholder. The real Home screen is T-203i.
struct HomeView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.sm) {
                    SectionHeader(title: "Recently Decoded")
                    Text("Songs you decode will show up here.")
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.lg)
            }
            .background(Palette.canvas)
            .navigationTitle("Home")
        }
    }
}

#Preview {
    HomeView()
        .preferredColorScheme(.dark)
}
