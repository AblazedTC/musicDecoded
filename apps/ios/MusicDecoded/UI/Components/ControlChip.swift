import MusicDecodedCore
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
        .buttonStyle(PushButtonStyle())
        .accessibilityLabel(title)
        .accessibilityValue(ChordDisplay.spoken(value))
    }
}

/// The look of a `ControlChip`, for use as the label of a `Menu` (e.g. Transpose −6…+6): a square
/// `surface` chip with a 3pt `line` border, a caption and the value.
struct ControlChipLabel: View {
    let title: String
    let value: String

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(title)
                    .chipCaptionStyle()
                    .lineLimit(1)
                Text(value)
                    .font(.rowTitle)
                    .monospacedDigit()
                    .foregroundStyle(Palette.text)
                    .lineLimit(dynamicTypeSize.isAccessibilitySize ? 3 : 1)
            }
            Spacer(minLength: Spacing.xxs)
            Image(systemName: "chevron.down")
                .font(.caption2.weight(.bold))
                .foregroundStyle(Palette.lime)
                .accessibilityHidden(true)
        }
        .padding(Spacing.sm)
        .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
        .inkCard()
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
