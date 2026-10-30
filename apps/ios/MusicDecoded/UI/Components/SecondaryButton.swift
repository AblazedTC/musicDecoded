import SwiftUI

/// Full-width secondary action ("Choose another file"): `surface` fill, white text, white-20% 3pt
/// border, no shadow.
struct SecondaryButton: View {
    let title: String
    var systemImage: String?
    let action: () -> Void

    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.xs) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .accessibilityHidden(true)
                }
                Text(title)
            }
            .font(.label)
            .foregroundStyle(isEnabled ? Palette.text : Palette.textMuted)
            .padding(.horizontal, Spacing.md)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.primaryButtonHeight)
            .inkCard()
            .contentShape(.rect)
        }
        .buttonStyle(PushButtonStyle())
    }
}

#Preview {
    SecondaryButton(title: "Choose another file") {}
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
