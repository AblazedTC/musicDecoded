import Testing
@testable import MusicDecodedCore

struct ChordVoicingsTests {
    let voicings: ChordVoicings

    init() throws {
        voicings = try Contracts.voicings()
    }

    @Test func matchesEveryExampleInTheContract() throws {
        let examples = try #require(voicings.examples)
        try #require(examples.isEmpty == false)
        for (label, expected) in examples {
            #expect(voicings.notes(for: label) == expected, "\(label)")
        }
    }

    @Test(arguments: [
        ("C:maj", [48, 52, 55]),
        ("Bb:7", [58, 62, 65, 68]),
        ("E:min/b3", [43, 52, 55, 59]),
        ("A:sus4(b7)", [57, 62, 64, 67]),
        ("C:min(9)", [48, 51, 55]),       // unknown → text before "(" → min
        ("C:weird", [48, 52, 55]),        // unknown → maj
        ("X", []),
        ("N", []),
    ])
    func followsTheAlgorithm(label: String, expected: [Int]) {
        #expect(voicings.notes(for: label) == expected)
    }

    @Test func transposeShiftsThePitchClassAndWraps() {
        #expect(voicings.notes(for: "A:maj", transpose: 3) == voicings.notes(for: "C:maj"))
        #expect(voicings.notes(for: "B:maj", transpose: 1) == [48, 52, 55])
        #expect(voicings.notes(for: "C:maj", transpose: -1) == [59, 63, 66])
        #expect(voicings.notes(for: "A:maj", transpose: -15) == voicings.notes(for: "F#:maj"))
        #expect(voicings.notes(for: "D:maj/3", transpose: 2) == [44, 52, 56, 59])
    }
}
