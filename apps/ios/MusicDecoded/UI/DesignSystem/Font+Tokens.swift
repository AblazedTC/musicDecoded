import SwiftUI

/// The bundled faces (DESIGN.md §1): Archivo Black (display) and Space Grotesk 400/500/700 (body),
/// SIL OFL (Resources/Licenses), registered with `UIAppFonts` in project.yml. These are the
/// fonts' PostScript names.
enum FontName {
    static let display = "ArchivoBlack-Regular"
    static let regular = "SpaceGrotesk-Regular"
    static let medium = "SpaceGrotesk-Medium"
    static let bold = "SpaceGrotesk-Bold"
}

/// Type roles from DESIGN.md §1. Every role is `Font.custom(_:size:relativeTo:)`, so it scales
/// with Dynamic Type; nothing is below 11pt at the default size. Display roles are uppercase:
/// apply `.textCase(.uppercase)` (or `DisplayHeadline`) at the call site.
extension Font {
    /// Page title: Archivo Black 44.
    static var pageTitle: Font { .custom(FontName.display, size: 44, relativeTo: .largeTitle) }
    /// Section title: Archivo Black 28.
    static var sectionHeader: Font { .custom(FontName.display, size: 28, relativeTo: .title) }
    /// The big chord name in the player: Archivo Black 56, lime on ink.
    static var currentChord: Font { .custom(FontName.display, size: 56, relativeTo: .largeTitle) }
    /// Archivo Black at any size (chord blocks, counters, hero numbers).
    static func display(_ size: CGFloat, relativeTo style: Font.TextStyle = .title) -> Font {
        .custom(FontName.display, size: size, relativeTo: style)
    }

    /// Card / row title: Space Grotesk Bold 17.
    static var rowTitle: Font { .custom(FontName.bold, size: 17, relativeTo: .headline) }
    /// Body: Space Grotesk Regular 15.
    static var bodyText: Font { .custom(FontName.regular, size: 15, relativeTo: .subheadline) }
    /// Secondary text, times, counters, BPM: Space Grotesk Medium 13 (add `.monospacedDigit()`
    /// for numbers).
    static var detail: Font { .custom(FontName.medium, size: 13, relativeTo: .footnote) }
    /// Buttons and chips: Space Grotesk Bold 15.
    static var label: Font { .custom(FontName.bold, size: 15, relativeTo: .subheadline) }
    /// Small links and emphasized secondary text: Space Grotesk Bold 13.
    static var labelSmall: Font { .custom(FontName.bold, size: 13, relativeTo: .footnote) }
    /// Eyebrow / caption: Space Grotesk Bold 11 (uppercase, wide tracking via `overlineStyle()`).
    static var overline: Font { .custom(FontName.bold, size: 11, relativeTo: .caption2) }
    /// Sticker label: Space Grotesk Bold 14.
    static var sticker: Font { .custom(FontName.bold, size: 14, relativeTo: .subheadline) }
}

/// Letter spacing in points for the 11pt eyebrow (+0.25em), chip captions (+0.1em) and stickers
/// (+0.04em at 14pt).
enum Tracking {
    static let eyebrow: CGFloat = 2.75
    static let chip: CGFloat = 1.1
    static let sticker: CGFloat = 0.56
    /// Display headlines: -0.01em at 44pt.
    static let display: CGFloat = -0.44
}

extension View {
    /// Uppercase caption with wide tracking, used above values ("KEY", "SPEED"). `textMuted` by
    /// default; pass `Palette.lime` for an eyebrow on ink or `Palette.white` on violet.
    func overlineStyle(_ color: Color = Palette.textMuted, tracking: CGFloat = Tracking.eyebrow) -> some View {
        font(.overline)
            .textCase(.uppercase)
            .tracking(tracking)
            .foregroundStyle(color)
    }

    /// The caption inside narrow chips (Key · Transpose…): +0.1em, because +0.25em breaks
    /// "TRANSPOSE" mid-word in a third-width chip.
    func chipCaptionStyle(_ color: Color = Palette.textSecondary) -> some View {
        overlineStyle(color, tracking: Tracking.chip)
    }
}
