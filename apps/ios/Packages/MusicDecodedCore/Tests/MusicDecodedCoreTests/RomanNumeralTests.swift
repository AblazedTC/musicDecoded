import Foundation
import Testing
@testable import MusicDecodedCore

/// Runs every case in `contracts/fixtures/roman-numerals.json` (shared with Android).
struct RomanNumeralTests {
    private struct Fixture: Decodable {
        struct Case: Decodable {
            let key: Analysis.Key?
            let label: String
            let numeral: String?
        }

        let cases: [Case]
    }

    @Test func everySharedCase() throws {
        let data = try Data(contentsOf: Contracts.url("fixtures/roman-numerals.json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        #expect(fixture.cases.isEmpty == false)
        for item in fixture.cases {
            let keyName = item.key.map { "\($0.tonic) \($0.mode.rawValue)" } ?? "no key"
            #expect(romanNumeral(for: item.label, key: item.key) == item.numeral, "\(item.label) in \(keyName)")
        }
    }

    @Test func transposeAndBasicDoNotChangeTheNumeral() {
        let key = Analysis.Key(tonic: "C", mode: .major)
        #expect(ChordDisplay(key: key, transpose: 3).numeral("G:7") == "V")
        #expect(ChordDisplay(key: key, basic: true).numeral("B:hdim7") == "viiø")
        #expect(ChordDisplay(key: key).numeral("X") == nil)
    }

    @Test(arguments: [
        ("III", "three"),
        ("i", "one"),
        ("♭VII", "flat seven"),
        ("vii°", "seven diminished"),
        ("viiø", "seven half-diminished"),
        ("III+", "three augmented"),
        ("♯iv", "sharp four"),
    ])
    func spoken(numeral: String, expected: String) {
        #expect(spokenRomanNumeral(numeral) == expected)
    }
}
