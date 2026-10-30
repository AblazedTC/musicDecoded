import SwiftUI

/// Placeholder artwork: the title's first letter on `surfaceSoft`, in `secondary`. The mini
/// player, Processing and song rows without a thumbnail all use it, so a song looks the same
/// everywhere (T-213).
struct InitialArt: View {
    let title: String
    var size: CGFloat = ComponentSize.songArt
    var cornerRadius: CGFloat = Radius.art
    var font: Font = .title3.weight(.semibold)

    var body: some View {
        Text(Self.initial(of: title))
            .font(font)
            .foregroundStyle(Palette.secondary)
            .lineLimit(1)
            .minimumScaleFactor(0.5)
            .frame(width: size, height: size)
            .background(Palette.surfaceSoft, in: .rect(cornerRadius: cornerRadius))
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
