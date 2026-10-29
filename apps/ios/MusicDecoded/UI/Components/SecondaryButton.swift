import SwiftUI

/// Full-width outlined button for the alternative action ("Choose another file").
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
            .font(.body.weight(.semibold))
            .foregroundStyle(Palette.text)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.primaryButtonHeight)
            .background(Palette.surface, in: .rect(cornerRadius: Radius.button))
            .overlay {
                RoundedRectangle(cornerRadius: Radius.button).strokeBorder(Palette.line, lineWidth: 1)
            }
            .opacity(isEnabled ? 1 : 0.5)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    SecondaryButton(title: "Choose another file") {}
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
