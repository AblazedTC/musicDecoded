import SwiftUI

/// A read-only metadata chip in the player header (Key · Time signature · BPM).
struct InfoChip: View {
    let title: String
    let value: String
    let systemImage: String

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xxs) {
            Label(title, systemImage: systemImage)
                .overlineStyle()
            Text(value)
                .font(.rowTitle)
                .monospacedDigit()
                .foregroundStyle(Palette.text)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.sm)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.control))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.control).strokeBorder(Palette.line, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
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
