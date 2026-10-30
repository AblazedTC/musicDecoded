/// Major-key numerals by semitones above the tonic (DESIGN.md §4 "Roman numerals").
private let majorNumerals = ["I", "♭II", "II", "♭III", "III", "IV", "♯IV", "V", "♭VI", "VI", "♭VII", "VII"]
/// Minor-key numerals, relative to the natural minor scale.
private let minorNumerals = ["I", "♭II", "II", "III", "♯III", "IV", "♯IV", "V", "VI", "♯VI", "VII", "♯VII"]

/// The Roman numeral of `label` in `key` (DESIGN.md §4): `E:maj` in C♯ minor → `III`,
/// `B:hdim7` in C major → `viiø`. Uses the chord root (not the slash bass); case follows the
/// quality; no extensions. `nil` for `N`/`X`, an unparseable label, or no key.
/// Transpose shifts the key and the chord equally, so it never changes the numeral.
public func romanNumeral(for label: String, key: Analysis.Key?) -> String? {
    guard let key,
          let tonic = PitchSpelling.pitchClass(of: key.tonic),
          let chord = ChordLabel(label),
          let root = PitchSpelling.pitchClass(of: chord.root)
    else { return nil }
    let degree = wrapPitchClass(root - tonic)
    let numeral = (key.mode == .minor ? minorNumerals : majorNumerals)[degree]
    let quality = chord.quality
    if quality.hasPrefix("hdim") { return numeral.lowercased() + "ø" }
    if quality.hasPrefix("dim") { return numeral.lowercased() + "°" }
    if quality.hasPrefix("min") { return numeral.lowercased() }
    if quality.hasPrefix("aug") { return numeral + "+" }
    return numeral
}

/// VoiceOver words for a numeral: `III` → "three", `♭VII` → "flat seven",
/// `vii°` → "seven diminished".
public func spokenRomanNumeral(_ numeral: String) -> String {
    let words = ["i": "one", "ii": "two", "iii": "three", "iv": "four", "v": "five", "vi": "six", "vii": "seven"]
    var prefix = ""
    var suffix = ""
    var core = numeral
    if core.hasPrefix("♭") {
        prefix = "flat "
        core.removeFirst()
    } else if core.hasPrefix("♯") {
        prefix = "sharp "
        core.removeFirst()
    }
    if core.hasSuffix("°") {
        suffix = " diminished"
        core.removeLast()
    } else if core.hasSuffix("ø") {
        suffix = " half-diminished"
        core.removeLast()
    } else if core.hasSuffix("+") {
        suffix = " augmented"
        core.removeLast()
    }
    guard let word = words[core.lowercased()] else { return numeral }
    return prefix + word + suffix
}

extension ChordDisplay {
    /// The Roman numeral of `label` in the analysis key, or `nil` (DESIGN.md §4). Not affected
    /// by Transpose or Chords: Basic.
    public func numeral(_ label: String) -> String? {
        romanNumeral(for: label, key: key)
    }
}
