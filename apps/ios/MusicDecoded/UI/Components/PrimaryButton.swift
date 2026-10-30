import SwiftUI

/// Full-width primary action ("Import & decode"), 48 tall: lime fill, ink text, 3pt black border
/// and a hard shadow (violet on ink by default; pass `Palette.border` on color blocks), with the
/// pushed-sticker pressed state. Disabled = `surfaceRaised`, muted text, no shadow.
struct PrimaryButton: View {
    let title: String
    var systemImage: String?
    var fill: Color = Palette.primary
    var shadowColor: Color = Palette.violet
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
            .foregroundStyle(isEnabled ? Palette.onColor : Palette.textMuted)
            .padding(.horizontal, Spacing.md)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.primaryButtonHeight)
            .contentShape(.rect)
        }
        .buttonStyle(
            BlockButtonStyle(
                fill: isEnabled ? fill : Palette.surfaceRaised,
                shadow: isEnabled ? HardShadow.md : 0,
                shadowColor: shadowColor,
                border: isEnabled ? Palette.border : Palette.line
            )
        )
    }
}

#Preview {
    VStack(spacing: Spacing.lg) {
        PrimaryButton(title: "Import & decode", systemImage: "waveform") {}
        PrimaryButton(title: "Decode this song") {}
            .disabled(true)
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
