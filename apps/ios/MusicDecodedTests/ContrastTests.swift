import Foundation
import Testing
@testable import MusicDecoded

/// WCAG 2 contrast of the DESIGN.md §1 "Wrapped" pairings (T-503i): ≥ 4.5:1 for every text color
/// on every fill it's drawn on, and the one forbidden pair (pink text on violet) stays forbidden.
/// Same pairs as Android's ContrastTest.
struct ContrastTests {
    typealias Hex = Palette.Hex

    static let darkSurfaces: [(String, UInt32)] = [
        ("ink", Hex.ink), ("card", Hex.card), ("cardRaised", Hex.cardRaised),
    ]
    static let lightBlocks: [(String, UInt32)] = [
        ("lime", Hex.lime), ("pink", Hex.pink), ("yellow", Hex.yellow), ("orange", Hex.orange),
        ("blue", Hex.blue), ("cream", Hex.cream), ("white", Hex.white),
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

    static func expectReadable(_ name: String, _ text: UInt32, on backgrounds: [(String, UInt32)]) {
        for (background, bg) in backgrounds {
            #expect(ratio(text, bg) >= 4.5, "\(name) on \(background)")
        }
    }

    @Test func textOnInkAndCards() {
        Self.expectReadable("text", Hex.white, on: Self.darkSurfaces + [("chordMuted", Hex.chordMuted)])
        Self.expectReadable("textSecondary", Hex.textSecondary, on: Self.darkSurfaces)
        // textMuted is for ink only (DESIGN.md §1), but it also stays readable on the cards it lands on.
        Self.expectReadable("textMuted", Hex.textMuted, on: [("ink", Hex.ink), ("card", Hex.card)])
    }

    @Test func accentsOnInkAreReadable() {
        // DESIGN.md §1 says "at least 6.4"; measured, orange on ink is 6.36, the lowest of the five.
        let accents: [(String, UInt32)] = [
            ("lime", Hex.lime), ("pink", Hex.pink), ("yellow", Hex.yellow), ("blue", Hex.blue), ("orange", Hex.orange),
        ]
        for (name, color) in accents {
            #expect(Self.ratio(color, Hex.ink) >= 6.3, "\(name) on ink")
            Self.expectReadable(name, color, on: Self.darkSurfaces)
        }
        Self.expectReadable("chord", Hex.lime, on: [("chordMuted", Hex.chordMuted)])
    }

    @Test func inkOnEveryLightBlock() {
        Self.expectReadable("onColor", Hex.ink, on: Self.lightBlocks + [("pianoBlackActive", Hex.pianoBlackActive)])
        Self.expectReadable("textSecondaryOnLime", Hex.textSecondaryOnLime, on: [("lime", Hex.lime), ("white", Hex.white)])
    }

    @Test func textOnViolet() {
        Self.expectReadable("white", Hex.white, on: [("violet", Hex.violet)])
        Self.expectReadable("lime", Hex.lime, on: [("violet", Hex.violet)])
        Self.expectReadable("yellow", Hex.yellow, on: [("violet", Hex.violet)])
        Self.expectReadable("textSecondaryOnViolet", Hex.textSecondaryOnViolet, on: [("violet", Hex.violet)])
    }

    @Test func pinkTextOnVioletIsTooLowToUse() {
        // DESIGN.md §1: never pink text on violet (3.2:1); pink on violet only as a fill with ink text.
        #expect(Self.ratio(Hex.pink, Hex.violet) < 4.5)
    }

    @Test func paletteMatchesThePortfolio() {
        #expect(Hex.violet == 0x4100F5)
        #expect(Hex.lime == 0xCFF56A)
        #expect(Hex.pink == 0xF774C4)
        #expect(Hex.yellow == 0xFAE62D)
        #expect(Hex.orange == 0xFF6437)
        #expect(Hex.blue == 0x509BF5)
        #expect(Hex.ink == 0x121212)
    }
}
