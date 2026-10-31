import Foundation

/// The shared voicing table from `contracts/chord-voicings.json`.
/// Both apps use it so their piano views show the same notes.
public struct ChordVoicings: Decodable, Sendable, Equatable {
    public let rootOctaveBase: Int
    public let bassOctaveBase: Int
    public let pitchClass: [String: Int]
    public let qualities: [String: [Int]]
    public let degreeSemitones: [String: Int]
    /// Label → expected MIDI notes. Test cases for `notes(for:)`.
    public let examples: [String: [Int]]?

    public init(data: Data) throws {
        self = try JSONDecoder().decode(ChordVoicings.self, from: data)
    }

    public init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    /// MIDI notes for a chord label, following the algorithm in the file's `description`.
    /// `transpose` shifts the root's pitch class (wrapping within the octave), so
    /// `notes(for: "A:maj", transpose: 3) == notes(for: "C:maj")`.
    public func notes(for label: String, transpose: Int = 0) -> [Int] {
        guard let chord = ChordLabel(label), let pitch = pitchClass[chord.root] else { return [] }
        let root = wrapPitchClass(pitch + transpose)
        let intervals = qualities[chord.quality]
            ?? qualities[String(chord.quality.prefix { $0 != "(" })]
            ?? qualities["maj"]
            ?? []
        var notes = intervals.map { rootOctaveBase + root + $0 }
        if let bass = chord.bass, let degree = degreeSemitones[bass] {
            notes.insert(bassOctaveBase + (root + degree) % 12, at: 0)
        }
        return notes
    }
}
