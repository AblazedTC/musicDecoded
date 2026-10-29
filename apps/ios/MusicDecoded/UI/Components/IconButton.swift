import SwiftUI

/// A round icon-only button (header search/profile, favorite, expand).
/// The visual is 40 or 32pt; the tappable area is padded to 44pt.
struct IconButton: View {
    enum Size {
        case regular, small

        var diameter: CGFloat {
            switch self {
            case .regular: ComponentSize.iconButton
            case .small: ComponentSize.smallIconButton
            }
        }
    }

    /// Read by VoiceOver.
    let title: String
    let systemImage: String
    var isActive = false
    var size: Size = .regular
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(size == .regular ? Font.body : Font.footnote)
                .fontWeight(.semibold)
                .foregroundStyle(isActive ? Palette.primary : Palette.text)
                .frame(width: size.diameter, height: size.diameter)
                .background(isActive ? Palette.primarySoft : Palette.surface, in: .circle)
                .overlay {
                    Circle().strokeBorder(isActive ? Palette.primary : Palette.line, lineWidth: 1)
                }
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }
}

#Preview {
    HStack(spacing: Spacing.xs) {
        IconButton(title: "Search", systemImage: "magnifyingglass") {}
        IconButton(title: "Favorite", systemImage: "heart.fill", isActive: true) {}
        IconButton(title: "Expand", systemImage: "arrow.up.left.and.arrow.down.right", size: .small) {}
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
