import Foundation
import Testing
@testable import MusicDecodedCore

/// Runs every case in `contracts/fixtures/piano-voicings.json` (shared with Android).
struct PianoVoicingTests {
    private struct Fixture: Decodable {
        struct Expected: Decodable {
            let inversions: Int
            let rightHand: [Int]
            let bass: Int
            let rightHandFingers: [Int]
            let bassFinger: Int
        }

        struct Case: Decodable {
            let label: String
            let inversion: Int
            let transpose: Int
            let expected: Expected?
        }

        let cases: [Case]
    }

    let voicings: ChordVoicings

    init() throws {
        voicings = try Contracts.voicings()
    }

    @Test func everySharedCase() throws {
        let data = try Data(contentsOf: Contracts.url("fixtures/piano-voicings.json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        #expect(fixture.cases.isEmpty == false)
        for item in fixture.cases {
            let name = "\(item.label) inv \(item.inversion) transpose \(item.transpose)"
            let voicing = voicings.pianoVoicing(label: item.label, inversion: item.inversion, transpose: item.transpose)
            guard let expected = item.expected else {
                #expect(voicing == nil, "\(name)")
                continue
            }
            guard let actual = voicing else {
                Issue.record("\(name): no voicing")
                continue
            }
            #expect(actual.inversions == expected.inversions, "\(name)")
            #expect(actual.rightHand == expected.rightHand, "\(name)")
            #expect(actual.bass == expected.bass, "\(name)")
            #expect(actual.rightHandFingers == expected.rightHandFingers, "\(name)")
            #expect(actual.bassFinger == expected.bassFinger, "\(name)")
        }
    }

    @Test func everyQualityFitsTheFixedKeyboard() {
        for quality in voicings.qualities.keys {
            for root in voicings.pitchClass.keys {
                for inversion in 0..<4 {
                    let label = "\(root):\(quality)"
                    guard let voicing = voicings.pianoVoicing(label: label, inversion: inversion) else {
                        Issue.record("\(label): no voicing")
                        continue
                    }
                    #expect(voicing.rightHand.allSatisfy { PianoVoicing.keyboardRange.contains($0) }, "\(label)")
                    #expect((48...59).contains(voicing.bass), "\(label)")
                    #expect(voicing.rightHand == voicing.rightHand.sorted(), "\(label)")
                    #expect(voicing.rightHand.count == voicing.rightHandFingers.count, "\(label)")
                }
            }
        }
    }

    @Test func negativeInversionWraps() {
        #expect(voicings.pianoVoicing(label: "C:maj", inversion: -1)?.inversion == 2)
    }

    @Test func displayClampsInsteadOfWrapping() {
        let display = ChordDisplay(key: nil, voicings: voicings)
        #expect(display.pianoVoicing("C:maj", inversion: 3)?.inversion == 2)
        #expect(display.pianoVoicing("G:5", inversion: 3)?.inversion == 1)
        #expect(display.pianoVoicing("C:7", inversion: 3)?.inversion == 3)
        #expect(display.pianoVoicing("C:maj", inversion: -2)?.inversion == 0)
        #expect(display.pianoVoicing("N", inversion: 0) == nil)
    }

    @Test func displayUsesTransposeAndBasic() {
        let display = ChordDisplay(key: nil, transpose: 2, basic: true, voicings: voicings)
        // C:maj7/3 → Basic C:maj → +2 → D major, no slash bass.
        let voicing = display.pianoVoicing("C:maj7/3", inversion: 0)
        #expect(voicing?.rightHand == [62, 66, 69])
        #expect(voicing?.bass == 50)
    }
}
