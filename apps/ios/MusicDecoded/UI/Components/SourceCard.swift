import SwiftUI

/// A card in the "Transcribe Music" grid: a color block in the source's fill (ink text, black
/// border, violet hard shadow on ink) with a white icon tile, the title and a one-line description.
/// Optionally tilted like a sticker.
struct SourceCard: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: SourceTint
    var tilt: Double = 0
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                SourceIconTile(systemImage: systemImage, fill: Palette.white)

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(title)
                        .font(.rowTitle)
                        .textCase(.uppercase)
                    Text(subtitle)
                        .font(.detail)
                        .multilineTextAlignment(.leading)
                }
                .foregroundStyle(Palette.onColor)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(Spacing.md)
            .contentShape(.rect)
        }
        .buttonStyle(BlockButtonStyle(fill: tint.fill, shadow: HardShadow.md, shadowColor: Palette.violet))
        .tilt(tilt)
    }
}

/// A source's icon tile: an ink icon on the source fill (or white) with a 3pt black border.
struct SourceIconTile: View {
    let systemImage: String
    var fill: Color

    var body: some View {
        Image(systemName: systemImage)
            .font(.title3.weight(.bold))
            .foregroundStyle(Palette.onColor)
            .frame(width: ComponentSize.sourceIconTile, height: ComponentSize.sourceIconTile)
            .block(fill)
            .accessibilityHidden(true)
    }
}

#Preview {
    Grid(horizontalSpacing: Spacing.md, verticalSpacing: Spacing.md) {
        GridRow {
            SourceCard(title: "Files", subtitle: "Import audio from this device", systemImage: "folder.fill", tint: .files, tilt: -Tilt.listCard) {}
            SourceCard(title: "Media Library", subtitle: "Choose local music", systemImage: "music.note.list", tint: .mediaLibrary, tilt: Tilt.listCard) {}
        }
        GridRow {
            SourceCard(title: "YouTube", subtitle: "Discover & play", systemImage: "play.rectangle.fill", tint: .youtube, tilt: Tilt.listCard) {}
            SourceCard(title: "SoundCloud", subtitle: "Find supported tracks", systemImage: "cloud.fill", tint: .soundcloud, tilt: -Tilt.listCard) {}
        }
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
