import SwiftUI

/// A square filter chip (All songs · Favorites · Offline · Recordings; YouTube recent searches).
/// On ink: selected = lime block with a violet shadow. With `onColorBlock` (the Library's lime
/// header): white chips with black borders, selected = violet with white text and a black shadow.
struct FilterChip: View {
    let title: String
    let isSelected: Bool
    var onColorBlock = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.sticker)
                .textCase(.uppercase)
                .tracking(Tracking.sticker)
                .foregroundStyle(foreground)
                .lineLimit(1)
                .padding(.horizontal, Spacing.sm)
                .padding(.vertical, Spacing.xs)
                .block(
                    fill,
                    shadow: isSelected ? HardShadow.sm : 0,
                    shadowColor: onColorBlock ? Palette.border : Palette.violet,
                    border: isSelected || onColorBlock ? Palette.border : Palette.borderOnInk
                )
                .frame(minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        }
        .buttonStyle(PushButtonStyle())
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var fill: Color {
        switch (onColorBlock, isSelected) {
        case (true, true): Palette.violet
        case (true, false): Palette.white
        case (false, true): Palette.primary
        case (false, false): Palette.surface
        }
    }

    private var foreground: Color {
        switch (onColorBlock, isSelected) {
        case (true, true): Palette.white
        case (false, false): Palette.text
        default: Palette.onColor
        }
    }
}

#Preview {
    VStack(alignment: .leading, spacing: Spacing.md) {
        HStack(spacing: Spacing.xs) {
            FilterChip(title: "All songs", isSelected: true, onColorBlock: true) {}
            FilterChip(title: "Favorites", isSelected: false, onColorBlock: true) {}
        }
        .padding(Spacing.md)
        .background(Palette.lime)
        HStack(spacing: Spacing.xs) {
            FilterChip(title: "Recent", isSelected: true) {}
            FilterChip(title: "lofi piano", isSelected: false) {}
        }
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
