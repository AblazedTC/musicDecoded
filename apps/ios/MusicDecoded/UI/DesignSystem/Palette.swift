import SwiftUI

/// DESIGN.md §1 "Wrapped" color tokens: ink surfaces with loud color blocks. Dark-only in v1, so
/// each token is one color. Screens use the semantic names; the raw palette is for color blocks.
///
/// A namespace instead of `Color` statics because `Color.primary` and `Color.secondary`
/// already exist and mean something else.
enum Palette {
    /// The raw 24-bit values, shared with ContrastTests.
    enum Hex {
        static let violet: UInt32 = 0x4100F5
        static let lime: UInt32 = 0xCFF56A
        static let pink: UInt32 = 0xF774C4
        static let yellow: UInt32 = 0xFAE62D
        static let orange: UInt32 = 0xFF6437
        static let blue: UInt32 = 0x509BF5
        static let ink: UInt32 = 0x121212
        static let card: UInt32 = 0x1C1C1C
        static let cardRaised: UInt32 = 0x262626
        static let white: UInt32 = 0xFFFFFF
        static let cream: UInt32 = 0xF0EDE4
        static let black: UInt32 = 0x000000

        static let chordMuted: UInt32 = 0x2A2A2A
        static let chordMutedBorder: UInt32 = 0x3A3A3A
        static let textSecondary: UInt32 = 0xB3B3B3
        static let textSecondaryOnViolet: UInt32 = 0xD6CCFF
        static let textSecondaryOnLime: UInt32 = 0x3A3A3A
        static let textMuted: UInt32 = 0x8A8A8A
        static let pianoBlackActive: UInt32 = 0x9DBF2E
    }

    // Palette (exact portfolio values).
    static let violet = Color(hex: Hex.violet)
    static let lime = Color(hex: Hex.lime)
    static let pink = Color(hex: Hex.pink)
    static let yellow = Color(hex: Hex.yellow)
    static let orange = Color(hex: Hex.orange)
    static let blue = Color(hex: Hex.blue)
    static let ink = Color(hex: Hex.ink)
    static let card = Color(hex: Hex.card)
    static let cardRaised = Color(hex: Hex.cardRaised)
    static let white = Color(hex: Hex.white)
    static let cream = Color(hex: Hex.cream)
    static let black = Color(hex: Hex.black)

    // Surfaces
    static let canvas = ink
    static let surface = card
    static let surfaceRaised = cardRaised

    // Accents
    static let primary = lime
    static let onPrimary = ink
    static let secondary = pink
    static let onSecondary = ink

    // Chords
    static let chord = lime
    static let onChord = ink
    static let chordMuted = Color(hex: Hex.chordMuted)
    static let chordMutedBorder = Color(hex: Hex.chordMutedBorder)

    // Text
    /// Text on ink and violet. On lime/pink/yellow/orange/blue/cream use `onColor`.
    static let text = white
    /// Text and icons on every light color block (lime, pink, yellow, orange, blue, cream).
    static let onColor = ink
    static let textSecondary = Color(hex: Hex.textSecondary)
    static let textSecondaryOnViolet = Color(hex: Hex.textSecondaryOnViolet)
    static let textSecondaryOnLime = Color(hex: Hex.textSecondaryOnLime)
    /// Captions and inactive tabs, on ink only.
    static let textMuted = Color(hex: Hex.textMuted)

    // Borders
    /// 3pt border on color blocks and cards on color blocks.
    static let border = black
    /// 3pt border around thumbnails/art on ink: white at 20%.
    static let borderOnInk = Color(hex: Hex.white, opacity: 0.2)
    /// Separators and quiet outlines on ink (same as the inactive chord border).
    static let line = chordMutedBorder

    // States
    static let record = orange
    static let destructive = orange
    static let info = blue
    static let warning = yellow

    // Waveform
    static let waveIdle = Color(hex: 0x3A3A3A)
    static let wavePlayed = lime

    // Piano
    static let pianoWhiteKey = cream
    static let pianoBlackKey = ink
    static let pianoActiveWhiteKey = lime
    static let pianoActiveBlackKey = Color(hex: Hex.pianoBlackActive)
    /// The left-hand bass note (DESIGN.md §3.7 "Owner revision 2": blue; right hand stays lime).
    static let pianoBass = blue
}
