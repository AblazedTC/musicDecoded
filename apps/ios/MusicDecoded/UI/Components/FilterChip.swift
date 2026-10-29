import SwiftUI

/// A selectable pill used to filter a list (All songs · Favorites · Offline · Recordings).
struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.detail.weight(.semibold))
                .foregroundStyle(isSelected ? Palette.onPrimary : Palette.textSecondary)
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.xs)
                .background(isSelected ? Palette.primary : Palette.surface, in: .rect(cornerRadius: Radius.pill))
                .overlay {
                    RoundedRectangle(cornerRadius: Radius.pill)
                        .strokeBorder(isSelected ? Palette.primary : Palette.line, lineWidth: 1)
                }
                .frame(minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

#Preview {
    HStack(spacing: Spacing.xs) {
        FilterChip(title: "All songs", isSelected: true) {}
        FilterChip(title: "Favorites", isSelected: false) {}
        FilterChip(title: "Offline", isSelected: false) {}
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
