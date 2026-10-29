import SwiftUI

/// Icon and tile colors for each source (DESIGN.md §1, "Source-icon tints").
struct SourceTint: Sendable {
    let icon: Color
    let background: Color

    static let files = SourceTint(icon: Palette.primary, background: Palette.primarySoft)
    static let mediaLibrary = SourceTint(icon: Palette.secondary, background: Color(hex: 0x1D3153))
    static let youtube = SourceTint(icon: Color(hex: 0xFF8B8B), background: Color(hex: 0x452534))
    static let soundcloud = SourceTint(icon: Color(hex: 0xFFB56F), background: Color(hex: 0x432F26))
    static let record = SourceTint(icon: Color(hex: 0xFF9A9A), background: Color(hex: 0x412534))
}
