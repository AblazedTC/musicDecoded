import Foundation

/// Builds a song row's meta line: `duration · key · BPM`, e.g. "4:23 · D major · 136 BPM".
/// Missing parts are left out.
enum SongMetadata {
    static func line(durationSec: Double?, key: String?, bpm: Double?) -> String {
        var parts: [String] = []
        if let durationSec {
            parts.append(Duration.seconds(durationSec).formatted(.time(pattern: .minuteSecond)))
        }
        if let key {
            parts.append(key)
        }
        if let bpm {
            parts.append("\(Int(bpm.rounded())) BPM")
        }
        return parts.joined(separator: " · ")
    }
}
