import Foundation

/// Display suffix per Harte quality (DESIGN.md §4). `maj` has no suffix.
private let qualitySuffixes: [String: String] = [
    "maj": "", "min": "m", "dim": "°", "aug": "+",
    "sus2": "sus2", "sus4": "sus4", "sus4(b7)": "7sus4", "5": "5",
    "7": "7", "maj7": "maj7", "min7": "m7", "minmaj7": "m(maj7)",
    "dim7": "°7", "hdim7": "m7♭5", "maj6": "6", "min6": "m6",
    "9": "9", "maj9": "maj9", "min9": "m9",
    "11": "11", "min11": "m11",
    "13": "13", "maj13": "maj13", "min13": "m13",
]

/// The name shown in the player, e.g. `A:maj` → `A`, `F#:min7` → `F♯m7`, `D:maj/3` → `D/F♯`.
/// `N`/`X` → `""`.
///
/// - Parameters:
///   - transpose: semitones (−6…+6 in the UI); wraps around the octave.
///   - accidentals: ♯/♭ preference. `.auto` reads `key` (shifted by `transpose`).
///   - simplify: Chords: Basic. Reduces the label to its triad family (major, minor, °, +,
///     sus2/sus4, 5) and drops the slash bass.
public func chordDisplayName(
    _ label: String,
    transpose: Int = 0,
    accidentals: Accidentals = .auto,
    key: Analysis.Key? = nil,
    simplify: Bool = false
) -> String {
    guard let chord = ChordLabel(label), let rootPitch = PitchSpelling.pitchClass(of: chord.root) else {
        return ""
    }
    let spelling = PitchSpelling(accidentals: accidentals, key: key, transpose: transpose)
    let root = rootPitch + transpose
    let quality = simplify ? triadFamily(of: chord.quality) : chord.quality
    var name = spelling.name(of: root) + suffix(for: quality)
    if simplify == false, let bass = chord.bass, let degree = PitchSpelling.semitones(ofDegree: bass) {
        name += "/" + spelling.name(of: root + degree)
    }
    return name
}

private func suffix(for quality: String) -> String {
    if let suffix = qualitySuffixes[quality] { return suffix }
    let base = String(quality.prefix { $0 != "(" })
    if let suffix = qualitySuffixes[base] { return suffix }
    return quality.replacingOccurrences(of: "b", with: "♭").replacingOccurrences(of: "#", with: "♯")
}

/// Chords: Basic. maj/7/maj7/9… → maj, min/min7/min9… → min; keeps dim, aug, sus and 5.
private func triadFamily(of quality: String) -> String {
    if quality.hasPrefix("dim") || quality.hasPrefix("hdim") { return "dim" }
    if quality.hasPrefix("min") { return "min" }
    if quality.hasPrefix("aug") { return "aug" }
    if quality.hasPrefix("sus2") { return "sus2" }
    if quality.hasPrefix("sus4") { return "sus4" }
    if quality == "5" { return "5" }
    return "maj"
}
