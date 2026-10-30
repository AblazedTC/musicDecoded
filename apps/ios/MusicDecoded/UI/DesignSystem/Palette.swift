import SwiftUI

/// Color tokens from DESIGN.md §1. The app is dark-only in v1, so each token is one color.
///
/// A namespace instead of `Color` statics because `Color.primary` and `Color.secondary`
/// already exist and mean something else.
enum Palette {
    // Backgrounds
    static let ink = Color(hex: 0x07101C)
    static let canvas = Color(hex: 0x0B1422)
    static let surface = Color(hex: 0x111F30)
    static let surfaceRaised = Color(hex: 0x182A3E)
    static let surfaceSoft = Color(hex: 0x20354B)
    static let line = Color(hex: 0x2A4057)

    // Accents
    static let primary = Color(hex: 0x62DCC4)
    static let onPrimary = Color(hex: 0x08221E)
    static let primarySoft = Color(hex: 0x173D3D)
    static let secondary = Color(hex: 0x91AFFF)

    // Chords
    static let chord = Color(hex: 0xF2DFAF)
    static let onChord = Color(hex: 0x322B1D)
    static let chordMuted = Color(hex: 0x463F32)
    static let chordMutedBorder = Color(hex: 0x645943)

    // Text
    static let text = Color(hex: 0xF5F8FB)
    static let textSecondary = Color(hex: 0x9FB0C3)
    /// #8396AA, not the prototype's #63778C (3.3–4.1:1): ≥ 4.5:1 on every surface text sits on
    /// (T-503, matches Android and DESIGN.md T-506).
    static let textMuted = Color(hex: 0x8396AA)

    // Record
    static let record = Color(hex: 0xEC6478)
    static let recordRing = Color(hex: 0xB94D5E)

    // Player
    static let transportBackground = Color(hex: 0x152638)
    static let transportBorder = Color(hex: 0x30465D)
    static let miniPlayerBackground = Color(hex: 0x1B2D40)
    static let timelineBackground = Color(hex: 0x09101A)
    static let timelineBorder = Color(hex: 0x24384B)
    static let waveIdle = Color(hex: 0x4E6379, opacity: 0.72)
    static let wavePlayed = Color(hex: 0x62DCC4, opacity: 0.8)

    // Piano
    static let pianoWhiteKey = Color(hex: 0xE7EBEC)
    static let pianoWhiteKeyBorder = Color(hex: 0x8D999D)
    static let pianoBlackKey = Color(hex: 0x0C1218)
    static let pianoActiveWhiteKey = Color(hex: 0x9DE8DB)
    static let pianoActiveBlackKey = Color(hex: 0x48BEA8)
    static let pianoBackground = Color(hex: 0x0A111A)
}
