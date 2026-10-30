import Testing
@testable import MusicDecodedCore

struct ChordSpokenNameTests {
    let voicings: ChordVoicings

    init() throws {
        voicings = try Contracts.voicings()
    }

    @Test(arguments: [
        ("A:maj", "A major"),
        ("F#:min7", "F sharp minor 7"),
        ("D:maj/3", "D major over F sharp"),
        ("B:hdim7", "B half-diminished 7"),
        ("C:dim", "C diminished"),
        ("E:aug", "E augmented"),
        ("G:7", "G 7"),
        ("A:sus4(b7)", "A 7 suspended 4"),
        ("N", ""),
        ("X", ""),
    ])
    func spokenNames(_ label: String, _ expected: String) {
        let chords = ChordDisplay(key: nil, voicings: voicings)
        #expect(chords.spokenName(label) == expected)
    }

    @Test func basicAndTransposeApply() {
        let chords = ChordDisplay(key: nil, transpose: 3, basic: true, voicings: voicings)
        #expect(chords.spokenName("A:min7/b3") == "C minor")
    }

    @Test func flatKeysSayFlat() {
        let chords = ChordDisplay(key: nil, voicings: voicings, accidentals: .flat)
        #expect(chords.spokenName("A#:maj") == "B flat major")
    }

    @Test func pianoSummary() {
        let chords = ChordDisplay(key: nil, voicings: voicings)
        #expect(chords.accessibilitySummary("D:maj") == "D major: D, F sharp, A")
        #expect(chords.accessibilitySummary("N") == "No chord")
        #expect(chords.accessibilitySummary(nil) == "No chord")
        #expect(ChordDisplay(key: nil).accessibilitySummary("D:maj") == "D major")
    }

    @Test func spokenAccidentals() {
        #expect(ChordDisplay.spoken("F♯") == "F sharp")
        #expect(ChordDisplay.spoken("B♭") == "B flat")
        #expect(ChordDisplay.spoken("C") == "C")
    }
}
