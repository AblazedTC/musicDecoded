import SwiftUI

/// A tappable chip with a label above its current value and a chevron
/// (player controls: Chords · Voicing · Transpose).
struct ControlChip: View {
    let title: String
    let value: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ControlChipLabel(title: title, value: value)
        }
        .buttonStyle(.plain)
    }
}

/// The look of a `ControlChip`, for use as the label of a `Menu` (e.g. Transpose −6…+6).
struct ControlChipLabel: View {
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(title)
                    .overlineStyle()
                Text(value)
                    .font(.rowTitle)
                    .monospacedDigit()
                    .foregroundStyle(Palette.text)
                    .lineLimit(1)
            }
            Spacer(minLength: Spacing.xxs)
            Image(systemName: "chevron.down")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(Palette.textSecondary)
                .accessibilityHidden(true)
        }
        .padding(Spacing.sm)
        .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.control))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.control).strokeBorder(Palette.line, lineWidth: 1)
        }
        .contentShape(.rect)
    }
}

#Preview {
    HStack(spacing: Spacing.xs) {
        ControlChip(title: "Chords", value: "Basic") {}
        ControlChip(title: "Voicing", value: "Simple") {}
        ControlChip(title: "Transpose", value: "+2") {}
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
