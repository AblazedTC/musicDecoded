import MusicDecodedCore
import SwiftUI

/// A tappable chip with a label above its current value and a chevron
/// (player controls: Chords · Voicing · Transpose). `inline` puts the label and value on one
/// line where the width allows (the player's compact control row).
struct ControlChip: View {
    let title: String
    let value: String
    var inline = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ControlChipLabel(title: title, value: value, inline: inline)
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
    var inline = false

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            if inline {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
                        caption
                        valueText
                    }
                    VStack(alignment: .leading, spacing: 0) {
                        caption
                        valueText
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    caption
                    valueText
                }
            }
            Spacer(minLength: Spacing.xxs)
            Image(systemName: "chevron.down")
                .font(.caption2.weight(.bold))
                .foregroundStyle(Palette.lime)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, inline ? Spacing.xs : Spacing.sm)
        .padding(.vertical, inline ? Spacing.xxs : Spacing.sm)
        .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
        .inkCard()
        .contentShape(.rect)
    }

    private var caption: some View {
        Text(title)
            .chipCaptionStyle()
            .lineLimit(1)
    }

    private var valueText: some View {
        Text(value)
            .font(inline ? .label : .rowTitle)
            .monospacedDigit()
            .foregroundStyle(Palette.text)
            .lineLimit(dynamicTypeSize.isAccessibilitySize ? 3 : 1)
    }
}

#Preview {
    VStack(spacing: Spacing.md) {
        HStack(spacing: Spacing.xs) {
            ControlChip(title: "Chords", value: "Basic") {}
            ControlChip(title: "Voicing", value: "Simple") {}
            ControlChip(title: "Transpose", value: "+2") {}
        }
        HStack(spacing: Spacing.xs) {
            ControlChip(title: "Chords", value: "Basic", inline: true) {}
            ControlChip(title: "Voicing", value: "Simple", inline: true) {}
            ControlChip(title: "Transpose", value: "+2", inline: true) {}
        }
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
