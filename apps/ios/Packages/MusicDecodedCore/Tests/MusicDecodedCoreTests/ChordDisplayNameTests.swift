import Testing
@testable import MusicDecodedCore

struct ChordDisplayNameTests {
    /// DESIGN.md §4 display table, plus every quality the fixture uses.
    @Test(arguments: [
        ("A:maj", "A"),
        ("F#:min7", "F♯m7"),
        ("D:maj/3", "D/F♯"),
        ("B:hdim7", "Bm7♭5"),
        ("B:dim", "B°"),
        ("C:aug", "C+"),
        ("N", ""),
        ("X", ""),
        ("G:7", "G7"),
        ("C:maj7", "Cmaj7"),
        ("E:min", "Em"),
        ("A:min9", "Am9"),
        ("A:sus4(b7)", "A7sus4"),
        ("E:min/b3", "Em/G"),
        ("C:dim7", "C°7"),
        ("C:maj(9)", "C"),
    ])
    func fullNames(label: String, expected: String) {
        #expect(chordDisplayName(label) == expected)
    }

    /// Chords: Basic keeps only the triad family and drops the slash bass.
    @Test(arguments: [
        ("A:maj", "A"),
        ("G:7", "G"),
        ("C:maj7", "C"),
        ("D:9", "D"),
        ("F#:min7", "F♯m"),
        ("A:min9", "Am"),
        ("E:minmaj7", "Em"),
        ("B:hdim7", "B°"),
        ("C:dim7", "C°"),
        ("C:aug", "C+"),
        ("A:sus4(b7)", "Asus4"),
        ("D:sus2", "Dsus2"),
        ("D:maj/3", "D"),
        ("N", ""),
    ])
    func basicSimplification(label: String, expected: String) {
        #expect(chordDisplayName(label, simplify: true) == expected)
    }

    @Test func forcedAccidentals() {
        #expect(chordDisplayName("A#:maj", accidentals: .sharp) == "A♯")
        #expect(chordDisplayName("A#:maj", accidentals: .flat) == "B♭")
        #expect(chordDisplayName("Gb:min7", accidentals: .sharp) == "F♯m7")
        #expect(chordDisplayName("D:maj/3", accidentals: .flat) == "D/G♭")
    }

    @Test func autoAccidentalsFollowTheKey() {
        let fMajor = Analysis.Key(tonic: "F", mode: .major)
        let dMajor = Analysis.Key(tonic: "D", mode: .major)
        let gMinor = Analysis.Key(tonic: "G", mode: .minor)
        let eMinor = Analysis.Key(tonic: "E", mode: .minor)
        let cMajor = Analysis.Key(tonic: "C", mode: .major)

        #expect(chordDisplayName("A#:maj", key: fMajor) == "B♭")
        #expect(chordDisplayName("D#:maj", key: gMinor) == "E♭")
        #expect(chordDisplayName("F#:min", key: dMajor) == "F♯m")
        #expect(chordDisplayName("D#:dim", key: eMinor) == "D♯°")
        #expect(chordDisplayName("A#:maj", key: cMajor) == "B♭", "C major uses the common spelling")
        #expect(chordDisplayName("C#:maj", key: cMajor) == "C♯")
        #expect(chordDisplayName("A#:maj") == "B♭", "no key uses the common spelling")
    }

    @Test func autoAccidentalsUseTheTransposedKey() {
        let fMajor = Analysis.Key(tonic: "F", mode: .major)
        // F major (flats) + 2 = G major (sharps): C# + 2 is spelled D♯, not E♭.
        #expect(chordDisplayName("C#:maj", transpose: 2, key: fMajor) == "D♯")
        #expect(chordDisplayName("C#:maj", transpose: 0, key: fMajor) == "D♭")
    }

    @Test func transposeWrapsAround() {
        let cases: [(label: String, transpose: Int, expected: String)] = [
            ("B:maj", 1, "C"),
            ("C:maj", -1, "B"),
            ("A:min7", 15, "Cm7"),
            ("A:min7", -9, "Cm7"),
            ("D:maj/3", 2, "E/G♯"),
            ("G:7", 12, "G7"),
            ("N", 3, ""),
        ]
        for testCase in cases {
            let name = chordDisplayName(testCase.label, transpose: testCase.transpose, accidentals: .sharp)
            #expect(name == testCase.expected, "\(testCase.label) \(testCase.transpose)")
        }
    }

    @Test func everyFixtureLabelHasAName() throws {
        let analysis = try Contracts.sampleAnalysis()
        for chord in analysis.chords where isNoChord(chord.label) == false {
            let name = chordDisplayName(chord.label, key: analysis.key)
            #expect(name.isEmpty == false, "\(chord.label)")
            #expect(name.contains(":") == false, "\(chord.label) → \(name)")
        }
    }
}
