import SwiftUI

/// A square icon-only button (header search/profile, favorite, expand, transport).
/// The visual is 40 or 32pt; the tappable area is padded to 44pt. On ink it's a `surface` square
/// with a white-20% border; `container` puts it on a color block (e.g. white on violet) with an ink
/// icon and a black border. Active = lime square, ink icon.
struct IconButton: View {
    enum Size {
        case regular, small

        var side: CGFloat {
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
    /// A fill for buttons on color blocks (black border, ink icon).
    var container: Color?
    let action: () -> Void

    private var onColor: Bool { isActive || container != nil }

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(size == .regular ? Font.body.weight(.bold) : Font.footnote.weight(.bold))
                .foregroundStyle(onColor ? Palette.onColor : Palette.text)
                .frame(width: size.side, height: size.side)
                .block(
                    container ?? (isActive ? Palette.primary : Palette.surface),
                    border: onColor ? Palette.border : Palette.borderOnInk
                )
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        }
        .buttonStyle(PushButtonStyle())
        // A fixed-size control: the glyph stops growing at xxxLarge, and at accessibility sizes
        // a long press shows it (with its title) in the Large Content Viewer instead.
        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
        .accessibilityShowsLargeContentViewer {
            Label(title, systemImage: systemImage)
        }
        .accessibilityLabel(title)
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }
}

#Preview {
    HStack(spacing: Spacing.xs) {
        IconButton(title: "Search", systemImage: "magnifyingglass") {}
        IconButton(title: "Favorite", systemImage: "heart.fill", isActive: true) {}
        IconButton(title: "Expand", systemImage: "arrow.up.left.and.arrow.down.right", size: .small) {}
        IconButton(title: "Profile", systemImage: "person.fill", container: Palette.white) {}
    }
    .padding(Spacing.md)
    .background(Palette.violet)
    .preferredColorScheme(.dark)
}
