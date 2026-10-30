/// The notes and fingering the player's fixed C3–B5 keyboard shows for one chord
/// (DESIGN.md §4 "Piano voicing & inversions"). MIDI note numbers throughout.
public struct PianoVoicing: Sendable, Equatable {
    /// Number of inversions of this chord (= right-hand tone count).
    public let inversions: Int
    /// The inversion shown, `0..<inversions` (after wrapping).
    public let inversion: Int
    /// Right-hand notes, lowest first.
    public let rightHand: [Int]
    /// The left-hand bass note (octave 3: 48…59).
    public let bass: Int
    /// One finger (1 = thumb … 5 = little finger) per `rightHand` note.
    public let rightHandFingers: [Int]
    public let bassFinger: Int

    public init(inversions: Int, inversion: Int, rightHand: [Int], bass: Int, rightHandFingers: [Int], bassFinger: Int) {
        self.inversions = inversions
        self.inversion = inversion
        self.rightHand = rightHand
        self.bass = bass
        self.rightHandFingers = rightHandFingers
        self.bassFinger = bassFinger
    }

    /// The fixed keyboard range: C3 (48) to B5 (83).
    public static let keyboardRange = 48...83
    /// The lowest right-hand start: C4.
    static let rightHandFloor = 60
    static let bassOctaveBase = 48
    static let maxRightHandTones = 4
    static let perfectFifth = 7

    /// Fingers for the right hand (§4 step 6).
    static func fingers(count: Int, inversion: Int) -> [Int] {
        switch count {
        case 1: [1]
        case 2: [1, 5]
        case 3: inversion == 1 ? [1, 2, 5] : [1, 3, 5]
        case 4: [1, 2, 3, 5]
        default: Array(1...max(count, 1))
        }
    }
}

extension ChordVoicings {
    /// The fixed-keyboard voicing for `label` (DESIGN.md §4): at most 4 right-hand tones (the
    /// perfect fifth goes first), rotated left by `inversion` (wrapping modulo the tone count),
    /// stacked upward from C4, kept at or below B5; the bass (slash degree, else the root) in
    /// octave 3. `nil` for `N`/`X` and unknown roots.
    public func pianoVoicing(label: String, inversion: Int = 0, transpose: Int = 0) -> PianoVoicing? {
        guard let chord = ChordLabel(label), let pitch = pitchClass[chord.root] else { return nil }
        let root = wrapPitchClass(pitch + transpose)
        var intervals = qualities[chord.quality]
            ?? qualities[String(chord.quality.prefix { $0 != "(" })]
            ?? qualities["maj"]
            ?? []
        guard intervals.isEmpty == false else { return nil }
        if intervals.count > PianoVoicing.maxRightHandTones, let fifth = intervals.firstIndex(of: PianoVoicing.perfectFifth) {
            intervals.remove(at: fifth)
        }
        let tones = intervals.prefix(PianoVoicing.maxRightHandTones).map { wrapPitchClass(root + $0) }
        let count = tones.count
        let k = ((inversion % count) + count) % count
        let rotated = Array(tones[k...] + tones[..<k])

        var rightHand: [Int] = []
        for pitchClass in rotated {
            let floor = rightHand.last.map { $0 + 1 } ?? PianoVoicing.rightHandFloor
            rightHand.append(floor + wrapPitchClass(pitchClass - floor))
        }
        if let top = rightHand.last, top > PianoVoicing.keyboardRange.upperBound {
            rightHand = rightHand.map { $0 - 12 }
        }

        let bassDegree = chord.bass.flatMap { degreeSemitones[$0] } ?? 0
        return PianoVoicing(
            inversions: count,
            inversion: k,
            rightHand: rightHand,
            bass: PianoVoicing.bassOctaveBase + wrapPitchClass(root + bassDegree),
            rightHandFingers: PianoVoicing.fingers(count: count, inversion: k),
            bassFinger: 5
        )
    }
}
