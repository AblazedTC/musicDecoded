import SwiftUI

/// Placeholder artwork: the title's first letter in lime display type on a `surfaceRaised` square
/// with a white-20% 3pt border. The mini player, Processing and song rows without a thumbnail all
/// use it, so a song looks the same everywhere (T-213).
struct InitialArt: View {
    let title: String
    var size: CGFloat = ComponentSize.songArt
    var font: Font = .display(24, relativeTo: .title3)
    var border: Color = Palette.borderOnInk

    var body: some View {
        Text(Self.initial(of: title))
            .font(font)
            .foregroundStyle(Palette.lime)
            .lineLimit(1)
            .minimumScaleFactor(0.5)
            .frame(width: size, height: size)
            .background(Palette.surfaceRaised, in: .rect)
            .squareBorder(border)
            .accessibilityHidden(true)
    }

    /// The first letter or digit of `title`, uppercased; `♪` when there's none.
    static func initial(of title: String) -> String {
        guard let first = title.first(where: { $0.isLetter || $0.isNumber }) else { return "♪" }
        return String(first).uppercased()
    }
}

#Preview {
    HStack(spacing: Spacing.xs) {
        InitialArt(title: "Sunny Days")
        InitialArt(title: "  ")
        InitialArt(title: "(intro)", size: ComponentSize.iconButton)
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
