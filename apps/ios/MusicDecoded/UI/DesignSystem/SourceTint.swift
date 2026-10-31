import SwiftUI

/// The fill of each source's tile (DESIGN.md §1, "Source tiles"). The icon and text on it are
/// always `Palette.onColor` (ink).
struct SourceTint: Sendable {
    let fill: Color

    static let files = SourceTint(fill: Palette.lime)
    static let mediaLibrary = SourceTint(fill: Palette.blue)
    static let youtube = SourceTint(fill: Palette.orange)
    static let soundcloud = SourceTint(fill: Palette.yellow)
    static let record = SourceTint(fill: Palette.pink)
}
