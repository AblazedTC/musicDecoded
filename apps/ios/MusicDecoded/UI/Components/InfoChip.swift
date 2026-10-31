import MusicDecodedCore
import SwiftUI

/// A read-only metadata chip in the player header (Key · Time signature · BPM): lime icon, caption
/// and value on a square `surface` chip with a 3pt `line` border.
struct InfoChip: View {
    let title: String
    let value: String
    let systemImage: String

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
            Image(systemName: systemImage)
                .font(.footnote.weight(.bold))
                .foregroundStyle(Palette.lime)
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
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.sm)
        .inkCard()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(value == "—" ? "Unknown" : ChordDisplay.spoken(value))
    }
}

#Preview {
    // Values from contracts/fixtures/sample.analysis.json.
    HStack(spacing: Spacing.xs) {
        InfoChip(title: "Key", value: "D major", systemImage: "music.note")
        InfoChip(title: "Time", value: "4/4", systemImage: "metronome")
        InfoChip(title: "Tempo", value: "136 BPM", systemImage: "waveform.path")
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
