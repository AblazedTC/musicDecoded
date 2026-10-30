/// Keyboard geometry in MIDI notes and white-key indices (white key 0 = MIDI 0, a C).
enum PianoLayout {
    static let whiteKeysPerOctave = 7
    /// Where the keyboard starts before any chord has played (C3).
    static let defaultStart = 48
    /// An 88-key piano: A0…C8.
    static let range = 21...108

    private static let blackPitchClasses: Set<Int> = [1, 3, 6, 8, 10]
    private static let whitePitchClasses = [0, 2, 4, 5, 7, 9, 11]
    private static let whiteLetters = ["C", "D", "E", "F", "G", "A", "B"]

    static func isBlackKey(_ midi: Int) -> Bool {
        blackPitchClasses.contains(floorMod(midi, 12))
    }

    /// White keys from MIDI 0 up to `midi` (a black key counts as the white key below it).
    static func whiteKeyIndex(_ midi: Int) -> Int {
        let pitchClass = floorMod(midi, 12)
        let inOctave = whitePitchClasses.lastIndex { $0 <= pitchClass } ?? 0
        return floorDiv(midi, 12) * whiteKeysPerOctave + inOctave
    }

    static func midiOfWhiteKey(_ index: Int) -> Int {
        floorDiv(index, whiteKeysPerOctave) * 12 + whitePitchClasses[floorMod(index, whiteKeysPerOctave)]
    }

    /// `C3` on Cs (middle C = C4), plain letters on the other white keys.
    static func whiteLabel(_ midi: Int) -> String {
        let letter = whitePitchClasses.firstIndex(of: floorMod(midi, 12)).map { whiteLetters[$0] } ?? ""
        return letter == "C" ? "C\(floorDiv(midi, 12) - 1)" : letter
    }

    /// The C (MIDI) the keyboard should start at so `notes` fit in `octaves` octaves; when they
    /// don't, the lowest notes (the slash bass first) are dropped until the rest fit. `nil` = no notes.
    static func keyboardStart(_ notes: [Int], octaves: Int) -> Int? {
        var candidates = notes.sorted()[...]
        while let lowest = candidates.first, let highest = candidates.last {
            let start = floorDiv(lowest, 12) * 12
            if highest < start + 12 * octaves {
                return start
            }
            candidates = candidates.dropFirst()
        }
        return nil
    }

    /// Narrowest white key the player's panel keyboard uses before it drops an octave.
    static let minWhiteKeyWidth = 28.0

    /// How many octaves fit `width` points with white keys at least `minWhiteKeyWidth` wide;
    /// never fewer than 2 (DESIGN.md §3.7 item 5).
    static func octavesThatFit(width: Double) -> Int {
        guard width.isFinite, width > 0 else { return 2 }
        return max(2, Int(width / (minWhiteKeyWidth * Double(whiteKeysPerOctave))))
    }

    /// The first white key (index) of an `octaves`-octave keyboard centered on `notes`, kept
    /// inside the 88 keys. Like `keyboardStart`, the lowest notes are dropped until the rest fit.
    /// `nil` = no notes.
    static func centeredFirstWhite(_ notes: [Int], octaves: Int) -> Int? {
        let whiteCount = octaves * whiteKeysPerOctave
        var candidates = notes.sorted()[...]
        while let lowest = candidates.first, let highest = candidates.last {
            let low = whiteKeyIndex(lowest)
            // A black key sits on the boundary with the next white key.
            let high = whiteKeyIndex(highest) + (isBlackKey(highest) ? 1 : 0)
            if high - low + 1 <= whiteCount {
                let first = floorDiv(low + high - whiteCount + 1, 2)
                let lowestFirst = whiteKeyIndex(range.lowerBound)
                let highestFirst = whiteKeyIndex(range.upperBound) - whiteCount + 1
                return min(max(first, lowestFirst), max(highestFirst, lowestFirst))
            }
            candidates = candidates.dropFirst()
        }
        return nil
    }

    private static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let quotient = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? quotient - 1 : quotient
    }

    private static func floorMod(_ a: Int, _ b: Int) -> Int {
        ((a % b) + b) % b
    }
}
