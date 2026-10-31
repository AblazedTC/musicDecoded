import SwiftUI

/// Rotating tips at the bottom of Home: a slightly tilted yellow block with ink text and a pink
/// hard shadow.
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
                .font(.title3.weight(.bold))
                .foregroundStyle(Palette.onColor)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(tip.title)
                    .font(.rowTitle)
                    .textCase(.uppercase)
                Text(tip.body)
                    .font(.detail)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(Palette.onColor)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
        }
        .padding(Spacing.md)
        .block(Palette.yellow, shadow: HardShadow.md, shadowColor: Palette.pink)
        .tilt(Tilt.listCard)
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
