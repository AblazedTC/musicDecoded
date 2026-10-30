import Foundation
import Testing

/// WCAG 2 contrast of the text tokens (T-503i) against the surfaces text is drawn on.
/// Values mirror `Palette` (DESIGN.md §1, with `textMuted` = #8396AA, see T-506).
struct ContrastTests {
    static let textTokens: [(String, UInt32)] = [
        ("text", 0xF5F8FB), ("textSecondary", 0x9FB0C3), ("textMuted", 0x8396AA),
        ("primary", 0x62DCC4), ("secondary", 0x91AFFF), ("chord", 0xF2DFAF), ("record", 0xEC6478),
    ]
    /// Backgrounds that carry body text (surfaceSoft/primarySoft carry only `text` and accents).
    static let backgrounds: [(String, UInt32)] = [
        ("ink", 0x07101C), ("canvas", 0x0B1422), ("surface", 0x111F30), ("surfaceRaised", 0x182A3E),
        ("transportBackground", 0x152638), ("timelineBackground", 0x09101A),
    ]

    static func luminance(_ hex: UInt32) -> Double {
        func channel(_ shift: UInt32) -> Double {
            let c = Double((hex >> shift) & 0xFF) / 255
            return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * channel(16) + 0.7152 * channel(8) + 0.0722 * channel(0)
    }

    static func ratio(_ a: UInt32, _ b: UInt32) -> Double {
        let (la, lb) = (luminance(a), luminance(b))
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    @Test func textTokensReachAA() {
        for (name, fg) in Self.textTokens {
            for (background, bg) in Self.backgrounds {
                #expect(Self.ratio(fg, bg) >= 4.5, "\(name) on \(background)")
            }
        }
    }

    @Test func filledControlsReachAA() {
        #expect(Self.ratio(0x08221E, 0x62DCC4) >= 4.5) // onPrimary on primary
        #expect(Self.ratio(0x322B1D, 0xF2DFAF) >= 4.5) // onChord on chord
        #expect(Self.ratio(0xF2DFAF, 0x463F32) >= 4.5) // chord name on chordMuted block
        #expect(Self.ratio(0xF5F8FB, 0x20354B) >= 4.5) // text on surfaceSoft (selected segment)
        #expect(Self.ratio(0xF5F8FB, 0x1B2D40) >= 4.5) // text on the mini player
        #expect(Self.ratio(0x9FB0C3, 0x1B2D40) >= 4.5) // textSecondary on the mini player
    }

    @Test func thePrototypeMutedFails() {
        #expect(Self.ratio(0x63778C, 0x111F30) < 4.5)
    }
}
