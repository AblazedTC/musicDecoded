import Testing
@testable import MusicDecodedCore

struct ChordDisplayTests {
    let voicings: ChordVoicings

    init() throws {
        voicings = try Contracts.voicings()
    }

    @Test func transposeChangesNamesNotesAndNoteNames() {
        let chords = ChordDisplay(key: nil, transpose: 3, voicings: voicings)
        #expect(chords.name("A:maj") == "C")
        #expect(chords.notes("A:maj") == voicings.notes(for: "C:maj"))
        #expect(chords.noteNames("A:maj") == "C · E · G")
    }

    @Test func basicSimplifiesNamesAndNotesAndDropsTheBass() {
        let full = ChordDisplay(key: nil, voicings: voicings)
        #expect(full.name("F#:min7") == "F♯m7")

        let basic = ChordDisplay(key: nil, basic: true, voicings: voicings)
        #expect(basic.name("F#:min7") == "F♯m")
        #expect(basic.notes("F#:min7") == voicings.notes(for: "F#:min"))
        #expect(basic.bassNote("D:maj/3") == nil)
    }

    @Test func slashBassIsTheLowestNote() throws {
        let chords = ChordDisplay(key: nil, voicings: voicings)
        let notes = chords.notes("D:maj/3")
        let bass = try #require(chords.bassNote("D:maj/3"))
        #expect(bass == notes[0])
        #expect(notes[0] < notes[1])
        #expect(chords.bassNote("D:maj") == nil)
    }

    @Test func noChordHasNoNameOrNotes() {
        let chords = ChordDisplay(key: nil, voicings: voicings)
        #expect(chords.name("N") == "")
        #expect(chords.notes("N").isEmpty)
        #expect(chords.noteNames("X") == "")
    }

    @Test func withoutVoicingsThereAreNoNotes() {
        let chords = ChordDisplay(key: nil)
        #expect(chords.name("A:maj") == "A")
        #expect(chords.notes("A:maj").isEmpty)
    }

    @Test func noteNamesFollowTheKeySignature() {
        let fMajor = Analysis.Key(tonic: "F", mode: .major, confidence: nil)
        let chords = ChordDisplay(key: fMajor, voicings: voicings)
        #expect(chords.noteName(70) == "B♭")
        #expect(chords.noteNames("G:min") == "G · B♭ · D")
    }
}
