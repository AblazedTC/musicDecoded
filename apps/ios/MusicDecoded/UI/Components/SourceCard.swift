import SwiftUI

/// A card in the "Transcribe Music" grid: icon tile, title and a one-line description.
struct SourceCard: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: SourceTint
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                Image(systemName: systemImage)
                    .font(.title3)
                    .foregroundStyle(tint.icon)
                    .frame(width: ComponentSize.sourceIconTile, height: ComponentSize.sourceIconTile)
                    .background(tint.background, in: .rect(cornerRadius: Radius.control))
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(title)
                        .font(.rowTitle)
                        .foregroundStyle(Palette.text)
                    Text(subtitle)
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                        .multilineTextAlignment(.leading)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(Spacing.md)
            .background(
                LinearGradient(colors: [Palette.surfaceRaised, Palette.surface], startPoint: .top, endPoint: .bottom),
                in: .rect(cornerRadius: Radius.card)
            )
            .overlay {
                RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
            }
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    Grid(horizontalSpacing: Spacing.xs, verticalSpacing: Spacing.xs) {
        GridRow {
            SourceCard(title: "Files", subtitle: "Import audio from this device", systemImage: "folder", tint: .files) {}
            SourceCard(title: "Media Library", subtitle: "Choose local music", systemImage: "music.note.list", tint: .mediaLibrary) {}
        }
        GridRow {
            SourceCard(title: "YouTube", subtitle: "Discover & play", systemImage: "play.rectangle", tint: .youtube) {}
            SourceCard(title: "SoundCloud", subtitle: "Find supported tracks", systemImage: "cloud", tint: .soundcloud) {}
        }
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
