import SwiftUI

/// The full-width "Record Audio" card under the source grid.
struct RecordCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: "mic")
                    .font(.title3)
                    .foregroundStyle(SourceTint.record.icon)
                    .frame(width: ComponentSize.sourceIconTile, height: ComponentSize.sourceIconTile)
                    .background(SourceTint.record.background, in: .rect(cornerRadius: Radius.control))
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Record Audio")
                        .font(.rowTitle)
                        .foregroundStyle(Palette.text)
                    Text("Capture a live performance")
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "chevron.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Palette.textMuted)
                    .accessibilityHidden(true)
            }
            .padding(Spacing.md)
            .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
            .overlay {
                RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
            }
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    RecordCard {}
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
