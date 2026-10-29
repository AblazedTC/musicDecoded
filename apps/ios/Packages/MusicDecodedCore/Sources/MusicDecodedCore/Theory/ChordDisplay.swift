/// How chords are shown with the current Chords (Basic/Full) and Transpose settings, so the
/// timeline, Now Playing and the piano always agree (DESIGN.md §4).
public struct ChordDisplay: Sendable, Equatable {
    public let key: Analysis.Key?
    /// Semitones, −6…+6 in the UI. Display only (not audio pitch).
    public let transpose: Int
    /// Chords: Basic. Names and notes use the triad family, without the slash bass.
    public let basic: Bool
    public let voicings: ChordVoicings?
    /// Settings → Accidentals.
    public let accidentals: Accidentals

    private let spelling: PitchSpelling

    public init(
        key: Analysis.Key?,
        transpose: Int = 0,
        basic: Bool = false,
        voicings: ChordVoicings? = nil,
        accidentals: Accidentals = .auto
    ) {
        self.key = key
        self.transpose = transpose
        self.basic = basic
        self.voicings = voicings
        self.accidentals = accidentals
        spelling = PitchSpelling(accidentals: accidentals, key: key, transpose: transpose)
    }

    /// `F#:min7` → `F♯m7` (Full) or `F♯m` (Basic); `""` for `N`/`X`.
    public func name(_ label: String) -> String {
        chordDisplayName(label, transpose: transpose, accidentals: accidentals, key: key, simplify: basic)
    }

    /// MIDI notes from `chord-voicings.json`; Basic voices the triad family without the slash bass.
    public func notes(_ label: String) -> [Int] {
        voicings?.notes(for: voicedLabel(label), transpose: transpose) ?? []
    }

    /// The slash-chord bass (always the lowest voiced note), or `nil`.
    public func bassNote(_ label: String) -> Int? {
        // `notes(for:)` adds the bass (as the first note) only when its degree is in the table.
        guard let bass = ChordLabel(voicedLabel(label))?.bass, voicings?.degreeSemitones[bass] != nil else {
            return nil
        }
        return notes(label).first
    }

    /// `C`, `F♯`, `B♭`… spelled for the (transposed) key.
    public func noteName(_ midi: Int) -> String {
        spelling.name(of: midi)
    }

    /// `B · D · F♯` (each pitch class once, lowest first).
    public func noteNames(_ label: String) -> String {
        var seen: [String] = []
        for note in notes(label) {
            let name = noteName(note)
            if seen.contains(name) == false {
                seen.append(name)
            }
        }
        return seen.joined(separator: " · ")
    }

    private func voicedLabel(_ label: String) -> String {
        guard basic, let chord = ChordLabel(label) else { return label }
        return "\(chord.root):\(triadFamily(of: chord.quality))"
    }
}
