import Foundation

/// Builds a song row's meta line: `duration · key · BPM`, e.g. "4:23 · F♯ minor · 136 BPM".
/// Missing parts are left out.
enum SongMetadata {
    static func line(durationSec: Double?, key: String?, bpm: Double?) -> String {
        var parts: [String] = []
        if let durationSec {
            parts.append(Duration.seconds(durationSec).formatted(.time(pattern: .minuteSecond)))
        }
        if let key {
            parts.append(displayKey(key))
        }
        if let bpm {
            parts.append("\(Int(bpm.rounded())) BPM")
        }
        return parts.joined(separator: " · ")
    }

    /// The stored key (`F# minor`, `Bb major`) with real glyphs (`F♯ minor`, `B♭ major`).
    /// Only the tonic is touched.
    static func displayKey(_ key: String) -> String {
        guard let space = key.firstIndex(of: " ") else { return glyphs(key) }
        return glyphs(String(key[..<space])) + String(key[space...])
    }

    private static func glyphs(_ tonic: String) -> String {
        tonic.replacingOccurrences(of: "#", with: "♯").replacingOccurrences(of: "b", with: "♭")
    }
}
