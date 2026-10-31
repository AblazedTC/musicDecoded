import CoreGraphics

/// How the Timeline view shares its height (DESIGN.md §3.7 "Owner revision 2"): the piano panel
/// first (keyboard 110–150), the media panel takes the rest; under 120 the media panel collapses
/// to a 56-tall strip and the keyboard takes what's left (at least 90).
enum PlayerLayout {
    /// The piano panel minus its keys: chord row + gap, note-name row, page dots, vertical padding.
    static let pianoPanelChrome: CGFloat = ComponentSize.pianoPanelChordRow + Spacing.xxs
        + ComponentSize.pianoPanelNamesRow + ComponentSize.pianoPanelDotsRow + Spacing.xxs * 2

    struct Split: Equatable {
        /// Keyboard (keys only) height.
        var keyboard: CGFloat
        /// Media panel (or strip) height.
        var media: CGFloat
        var mediaCollapsed: Bool
    }

    /// Splits `available` points (piano panel + `gap` + media panel) between the two panels.
    static func split(available: CGFloat, gap: CGFloat) -> Split {
        let forKeys = available - pianoPanelChrome - gap
        let withMedia = forKeys - ComponentSize.mediaPanelMin
        if withMedia >= ComponentSize.fixedKeyboardMin {
            let keyboard = min(withMedia, ComponentSize.fixedKeyboardMax)
            return Split(keyboard: keyboard, media: forKeys - keyboard, mediaCollapsed: false)
        }
        let keyboard = min(max(forKeys - ComponentSize.mediaStripHeight, ComponentSize.fixedKeyboardCompact), ComponentSize.fixedKeyboardMax)
        return Split(keyboard: keyboard, media: ComponentSize.mediaStripHeight, mediaCollapsed: true)
    }

    /// The smallest piano panel + gap + media strip.
    static func minimumPanels(gap: CGFloat) -> CGFloat {
        pianoPanelChrome + ComponentSize.fixedKeyboardCompact + gap + ComponentSize.mediaStripHeight
    }
}
