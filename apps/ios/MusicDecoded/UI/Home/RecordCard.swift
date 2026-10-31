import SwiftUI

/// The full-width "Record Audio" card under the source grid: a pink block (the Record source fill)
/// with ink text, black border and a violet hard shadow.
struct RecordCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.sm) {
                SourceIconTile(systemImage: "mic.fill", fill: Palette.white)

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Record Audio")
                        .font(.rowTitle)
                        .textCase(.uppercase)
                    Text("Capture a live performance")
                        .font(.detail)
                }
                .foregroundStyle(Palette.onColor)
                .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "arrow.right")
                    .font(.body.weight(.bold))
                    .foregroundStyle(Palette.onColor)
                    .accessibilityHidden(true)
            }
            .padding(Spacing.md)
            .contentShape(.rect)
        }
        .buttonStyle(BlockButtonStyle(fill: SourceTint.record.fill, shadow: HardShadow.md, shadowColor: Palette.violet))
    }
}

#Preview {
    RecordCard {}
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
