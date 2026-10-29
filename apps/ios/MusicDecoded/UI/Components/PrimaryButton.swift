import SwiftUI

/// Full-width filled button for the main action on a screen ("Import & decode").
struct PrimaryButton: View {
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
            .foregroundStyle(Palette.onPrimary)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.primaryButtonHeight)
            .background(Palette.primary, in: .rect(cornerRadius: Radius.button))
            .opacity(isEnabled ? 1 : 0.5)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    VStack(spacing: Spacing.sm) {
        PrimaryButton(title: "Import & decode", systemImage: "waveform") {}
        PrimaryButton(title: "Decode this song") {}
            .disabled(true)
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
