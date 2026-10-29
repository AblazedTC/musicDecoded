import CoreGraphics

/// Fixed component sizes from DESIGN.md §1. These are sizes, not spacing.
enum ComponentSize {
    /// Smallest tappable area on iOS. Pad smaller visuals up to it.
    static let minTouchTarget: CGFloat = 44
    static let iconButton: CGFloat = 40
    static let smallIconButton: CGFloat = 32
    static let playButton: CGFloat = 56
    static let recordButton: CGFloat = 84
    static let songArt: CGFloat = 56
    static let sourceIconTile: CGFloat = 44
    static let primaryButtonHeight: CGFloat = 48
    static let searchFieldHeight: CGFloat = 44
}
