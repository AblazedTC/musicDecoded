import CoreGraphics

/// Corner radii (DESIGN.md §1, "Shape & elevation").
enum Radius {
    /// Cards, timeline, visualizer.
    static let card: CGFloat = 18
    /// Song rows, mini player, summary.
    static let row: CGFloat = 15
    /// Search field and other controls.
    static let control: CGFloat = 12
    static let button: CGFloat = 13
    static let chip: CGFloat = 10
    /// Filter chips.
    static let pill: CGFloat = 20
    /// Top corners of the transport and import sheets.
    static let sheet: CGFloat = 22
    /// Song artwork.
    static let art: CGFloat = 10
}
