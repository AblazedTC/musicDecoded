import MusicDecodedCore
import Testing
@testable import MusicDecoded

/// Ports apps/android/.../player/PianoKeyboardTest.kt.
struct PianoLayoutTests {
    @Test func blackKeys() {
        #expect((0...11).filter(PianoLayout.isBlackKey) == [1, 3, 6, 8, 10])
        #expect(PianoLayout.isBlackKey(61))
        #expect(PianoLayout.isBlackKey(60) == false)
    }

    @Test func whiteKeyIndexRoundTrips() {
        #expect(PianoLayout.whiteKeyIndex(60) == 35) // C4 = 5 octaves × 7
        #expect(PianoLayout.whiteKeyIndex(61) == 35) // C♯ sits on C's white key
        #expect(PianoLayout.whiteKeyIndex(71) == 41) // B4
        for index in 0...74 {
            #expect(PianoLayout.whiteKeyIndex(PianoLayout.midiOfWhiteKey(index)) == index)
        }
        #expect(PianoLayout.midiOfWhiteKey(35) == 60)
        #expect(PianoLayout.midiOfWhiteKey(38) == 65)
    }

    @Test func whiteLabels() {
        #expect(PianoLayout.whiteLabel(60) == "C4")
        #expect(PianoLayout.whiteLabel(48) == "C3")
        #expect(PianoLayout.whiteLabel(62) == "D")
    }

    @Test func startsAtTheCBelowTheLowestNote() {
        #expect(PianoLayout.keyboardStart([50, 54, 57], octaves: 2) == 48) // D3 F♯3 A3
        #expect(PianoLayout.keyboardStart([42, 50, 54, 57], octaves: 2) == 36) // D/F♯ fits from C2
    }

    @Test func dropsTheBassWhenTheChordWontFit() {
        // Bass C2 + chord up to E4: 2 octaves can't hold both, so the chord tones win.
        #expect(PianoLayout.keyboardStart([36, 48, 52, 64], octaves: 2) == 48)
        #expect(PianoLayout.keyboardStart([36, 48, 52, 64], octaves: 3) == 36)
    }

    @Test func noNotesHasNoStart() {
        #expect(PianoLayout.keyboardStart([], octaves: 2) == nil)
    }

    @Test func everyVoicingExampleFitsTwoOctavesWithoutItsBass() throws {
        let voicings = try PlayerFixtures.voicings()
        let examples = try #require(voicings.examples)
        for (label, notes) in examples where notes.isEmpty == false {
            let start = try #require(PianoLayout.keyboardStart(notes, octaves: 2), "\(label)")
            let chordTones = label.contains("/") ? Array(notes.dropFirst()) : notes
            #expect(chordTones.allSatisfy { (start..<(start + 24)).contains($0) }, "\(label)")
        }
    }
}
