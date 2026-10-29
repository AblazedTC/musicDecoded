import SwiftUI

/// Rotating tips (sparkle icon, chord color) at the bottom of Home.
struct TipCard: View {
    struct Tip: Sendable {
        let title: String
        let body: String
    }

    static let tips = [
        Tip(title: "Practice smarter", body: "Slow down any decoded song without changing its key."),
        Tip(title: "Loop a tricky bar", body: "Tap loop in the player to repeat the current bar until it clicks."),
        Tip(title: "Change the key", body: "Transpose the chords up or down to suit your voice or capo."),
        Tip(title: "Works offline", body: "Decoded songs with the download badge play without a connection."),
    ]

    private static let interval = Duration.seconds(8)

    @State private var index = 0

    var body: some View {
        let tip = Self.tips[index % Self.tips.count]
        HStack(alignment: .top, spacing: Spacing.sm) {
            Image(systemName: "sparkles")
                .font(.title3)
                .foregroundStyle(Palette.chord)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(tip.title)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.chord)
                Text(tip.body)
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
        }
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
        }
        .task {
            while true {
                do {
                    try await Task.sleep(for: Self.interval)
                } catch {
                    return
                }
                withAnimation {
                    index = (index + 1) % Self.tips.count
                }
            }
        }
    }
}

#Preview {
    TipCard()
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
