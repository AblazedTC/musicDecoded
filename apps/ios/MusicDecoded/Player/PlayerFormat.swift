import Foundation
import MusicDecodedCore

/// Text for player values. Times use tabular digits in the UI (`.monospacedDigit()`).
enum PlayerFormat {
    /// `m:ss`, e.g. `1:04`. Negative times read as `0:00`.
    static func time(_ seconds: Double) -> String {
        let total = seconds.isFinite ? max(Int(seconds.rounded(.down)), 0) : 0
        return "\(total / 60):" + (total % 60 < 10 ? "0" : "") + "\(total % 60)"
    }

    /// Transpose chip value: `0`, `+2`, `−3` (real minus sign).
    static func transpose(_ semitones: Int) -> String {
        if semitones > 0 { return "+\(semitones)" }
        if semitones < 0 { return "−\(-semitones)" }
        return "0"
    }

    /// `1.00×`.
    static func speed(_ speed: Double) -> String {
        let hundredths = Int((speed * 100).rounded())
        let fraction = hundredths % 100
        return "\(hundredths / 100)." + (fraction < 10 ? "0" : "") + "\(fraction)×"
    }

    /// `Looping 1:04–1:20`.
    static func loop(_ range: Range<Double>) -> String {
        "\(time(range.lowerBound))–\(time(range.upperBound))"
    }

    /// `D major` with a real ♯ (`F♯ minor`).
    static func key(_ key: Analysis.Key) -> String {
        "\(key.tonic.replacingOccurrences(of: "#", with: "♯")) \(key.mode.rawValue)"
    }

    /// `in 1 beat`, `in 4 beats`.
    static func inBeats(_ beats: Int) -> String {
        beats == 1 ? "in 1 beat" : "in \(beats) beats"
    }

    /// Voicing chip value: `Root`, `1st inv`, `2nd inv`, `3rd inv`.
    static func inversion(_ inversion: Int) -> String {
        switch inversion {
        case 1: "1st inv"
        case 2: "2nd inv"
        case 3: "3rd inv"
        default: "Root"
        }
    }

    /// Spoken inversion: `Root position`, `First inversion`…
    static func spokenInversion(_ inversion: Int) -> String {
        switch inversion {
        case 1: "First inversion"
        case 2: "Second inversion"
        case 3: "Third inversion"
        default: "Root position"
        }
    }

    /// `1 beat`, `4 beats`.
    static func beats(_ beats: Int) -> String {
        beats == 1 ? "1 beat" : "\(beats) beats"
    }
}
