/// Turns pitch classes into note names with real ♯/♭ glyphs.
struct PitchSpelling: Sendable {
    private static let sharpNames = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
    private static let flatNames = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
    private static let commonNames = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]

    /// Major keys (by tonic pitch class) whose signature uses flats: F B♭ E♭ A♭ D♭.
    private static let flatMajorTonics: Set<Int> = [5, 10, 3, 8, 1]

    private let names: [String]

    /// `transpose` also moves the key, so Auto spells for the key the user is reading in.
    init(accidentals: Accidentals, key: Analysis.Key?, transpose: Int) {
        switch accidentals {
        case .sharp:
            names = Self.sharpNames
        case .flat:
            names = Self.flatNames
        case .auto:
            guard let key, let tonic = Self.pitchClass(of: key.tonic) else {
                names = Self.commonNames
                return
            }
            // Compare keys through their relative major (A minor ↔ C major).
            let relativeMajor = wrapPitchClass(tonic + transpose + (key.mode == .minor ? 3 : 0))
            if relativeMajor == 0 {
                names = Self.commonNames
            } else if Self.flatMajorTonics.contains(relativeMajor) {
                names = Self.flatNames
            } else {
                names = Self.sharpNames
            }
        }
    }

    func name(of pitchClass: Int) -> String {
        names[wrapPitchClass(pitchClass)]
    }

    /// Pitch class of a note name such as `C`, `F#`, `Bb` (ASCII accidentals, as in labels).
    static func pitchClass(of note: String) -> Int? {
        let naturals: [Character: Int] = ["C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11]
        guard let letter = note.first, let natural = naturals[letter] else { return nil }
        var value = natural
        for accidental in note.dropFirst() {
            switch accidental {
            case "#": value += 1
            case "b": value -= 1
            default: return nil
            }
        }
        return wrapPitchClass(value)
    }

    /// Semitones above the root for a Harte bass degree such as `3`, `b3`, `#4`, `b9`.
    static func semitones(ofDegree degree: String) -> Int? {
        let majorScale = [0, 2, 4, 5, 7, 9, 11]
        var shift = 0
        var digits = Substring(degree)
        while let first = digits.first, first == "b" || first == "#" {
            shift += first == "b" ? -1 : 1
            digits = digits.dropFirst()
        }
        guard let number = Int(digits), number >= 1 else { return nil }
        return majorScale[(number - 1) % 7] + 12 * ((number - 1) / 7) + shift
    }
}
