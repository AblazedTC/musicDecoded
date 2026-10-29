import SwiftUI

/// Home's header: the tilted logo mark, wordmark and tagline, plus Search and Profile.
struct BrandHeader: View {
    let onSearch: () -> Void
    let onProfile: () -> Void

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: "waveform")
                .font(.body.weight(.bold))
                .foregroundStyle(Palette.onPrimary)
                .frame(width: ComponentSize.iconButton, height: ComponentSize.iconButton)
                .background(Palette.primary, in: .rect(cornerRadius: Radius.control))
                .rotationEffect(.degrees(-4))
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 0) {
                Text("MusicDecoded")
                    .font(.sectionHeader)
                    .foregroundStyle(Palette.text)
                Text("Your music, decoded.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)

            // DESIGN.md §5: the header search opens Library search.
            IconButton(title: "Search your library", systemImage: "magnifyingglass", action: onSearch)
            IconButton(title: "Profile and settings", systemImage: "person", action: onProfile)
        }
    }
}

#Preview {
    BrandHeader(onSearch: {}, onProfile: {})
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
