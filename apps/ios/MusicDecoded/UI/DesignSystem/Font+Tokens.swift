import SwiftUI

/// Type roles from DESIGN.md §1, mapped onto system text styles so Dynamic Type works.
/// The smallest role is `.caption2` (11pt at the default size).
extension Font {
    static var pageTitle: Font { .largeTitle.bold() }
    /// The big chord name in the player: 34pt bold at the default size (DESIGN.md), via
    /// `.largeTitle` so it follows Dynamic Type (T-503i).
    static var currentChord: Font { .largeTitle.bold() }
    static var sectionHeader: Font { .title3.bold() }
    static var rowTitle: Font { .subheadline.weight(.semibold) }
    /// Body and secondary text.
    static var detail: Font { .footnote }
    /// Captions and overlines.
    static var overline: Font { .caption2 }
}

extension View {
    /// Uppercase caption with wide tracking, used above values ("NOW PLAYING", "KEY").
    func overlineStyle() -> some View {
        font(.overline.weight(.semibold))
            .textCase(.uppercase)
            .tracking(1)
            .foregroundStyle(Palette.textMuted)
    }
}
