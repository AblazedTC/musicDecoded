import SwiftUI

/// Home's violet hero block (DESIGN.md §1 color map): the tilted lime logo mark, a "Chord decoder"
/// sticker, Search and Profile, the "MUSIC DECODED" display wordmark, the tagline and lime waveform
/// bars on a halftone texture. Full-bleed: the caller doesn't pad it.
struct BrandHeader: View {
    let onSearch: () -> Void
    let onProfile: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            HStack(spacing: Spacing.md) {
                Image(systemName: "waveform")
                    .font(.body.weight(.bold))
                    .foregroundStyle(Palette.onColor)
                    .frame(width: ComponentSize.iconButton, height: ComponentSize.iconButton)
                    .block(Palette.lime, shadow: HardShadow.sm)
                    .tilt(Tilt.sticker)
                    .accessibilityHidden(true)

                Sticker(text: "Chord decoder", fill: Palette.pink)
                    .accessibilityHidden(true)

                Spacer(minLength: Spacing.xs)

                // DESIGN.md §5: the header search opens Library search.
                IconButton(title: "Search your library", systemImage: "magnifyingglass", container: Palette.white, action: onSearch)
                IconButton(title: "Profile and settings", systemImage: "person.fill", container: Palette.white, action: onProfile)
            }

            VStack(alignment: .leading, spacing: Spacing.xs) {
                VStack(alignment: .leading, spacing: 0) {
                    DisplayHeadline(text: "Music", font: .display(64, relativeTo: .largeTitle), color: Palette.white)
                    DisplayHeadline(text: "Decoded", font: .display(64, relativeTo: .largeTitle), color: Palette.lime)
                }
                .minimumScaleFactor(0.5)
                .lineLimit(1)
                Text("Your music, decoded.")
                    .font(.rowTitle)
                    .foregroundStyle(Palette.white)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("MusicDecoded. Your music, decoded.")
            .accessibilityAddTraits(.isHeader)

            WaveformBars(bars: 24, color: Palette.lime)
                .frame(height: 40)
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            HalftoneTexture()
        }
        .background(Palette.violet)
    }
}

#Preview {
    BrandHeader(onSearch: {}, onProfile: {})
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
