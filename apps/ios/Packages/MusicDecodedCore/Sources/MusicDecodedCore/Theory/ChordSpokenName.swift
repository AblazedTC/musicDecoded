/// Words for VoiceOver (T-503): "F♯m7" is read badly, so screens speak
/// "F sharp minor 7" instead, and the piano summary as "D major: D, F sharp, A".
private let spokenQualities: [String: String] = [
    "maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented",
    "sus2": "suspended 2", "sus4": "suspended 4", "sus4(b7)": "7 suspended 4", "5": "power chord",
    "7": "7", "maj7": "major 7", "min7": "minor 7", "minmaj7": "minor major 7",
    "dim7": "diminished 7", "hdim7": "half-diminished 7", "maj6": "6", "min6": "minor 6",
    "9": "9", "maj9": "major 9", "min9": "minor 9",
    "11": "11", "min11": "minor 11",
    "13": "13", "maj13": "major 13", "min13": "minor 13",
]

extension ChordDisplay {
    /// `F#:min7` → "F sharp minor 7", `D:maj/3` → "D major over F sharp"; `""` for `N`/`X`.
    /// Follows Transpose, Chords (Basic/Full) and Accidentals like `name(_:)`.
    public func spokenName(_ label: String) -> String {
        guard let chord = ChordLabel(label), let rootPitch = PitchSpelling.pitchClass(of: chord.root) else {
            return ""
        }
        let root = rootPitch + transpose
        let quality = basic ? triadFamily(of: chord.quality) : chord.quality
        var words = [Self.spoken(noteName(root)), Self.spokenQuality(quality)].filter { $0.isEmpty == false }
        if basic == false, let bass = chord.bass, let degree = PitchSpelling.semitones(ofDegree: bass) {
            words += ["over", Self.spoken(noteName(root + degree))]
        }
        return words.joined(separator: " ")
    }

    /// "D, F sharp, A" (each pitch class once, lowest first), or `""`.
    public func spokenNoteNames(_ label: String) -> String {
        noteNames(label)
            .components(separatedBy: " · ")
            .filter { $0.isEmpty == false }
            .map(Self.spoken)
            .joined(separator: ", ")
    }

    /// The piano / chord summary: "D major: D, F sharp, A", "D major" without voicings, or
    /// "No chord".
    public func accessibilitySummary(_ label: String?) -> String {
        guard let label else { return "No chord" }
        let name = spokenName(label)
        guard name.isEmpty == false else { return "No chord" }
        let notes = spokenNoteNames(label)
        return notes.isEmpty ? name : "\(name): \(notes)"
    }

    /// `F♯` → "F sharp", `B♭` → "B flat".
    public static func spoken(_ noteName: String) -> String {
        noteName
            .replacingOccurrences(of: "♯", with: " sharp")
            .replacingOccurrences(of: "♭", with: " flat")
    }

    private static func spokenQuality(_ quality: String) -> String {
        if let words = spokenQualities[quality] { return words }
        let base = String(quality.prefix { $0 != "(" })
        if let words = spokenQualities[base] { return words }
        return quality
            .replacingOccurrences(of: "b", with: " flat ")
            .replacingOccurrences(of: "#", with: " sharp ")
            .split(separator: " ")
            .joined(separator: " ")
    }
}
