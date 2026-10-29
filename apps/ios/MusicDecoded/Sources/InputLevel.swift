import Foundation

/// Input-level math for the recorder's meter and waveform (DESIGN.md §3.6).
enum InputLevel {
    static let segments = 16
    /// Bars in the live waveform.
    static let waveformBars = 48
    /// The meter's floor; quieter input reads as this.
    static let minDb: Float = -60

    /// A power reading in dBFS (AVAudioRecorder gives -160...0), clamped to `minDb...0`.
    static func clamp(_ db: Float) -> Float {
        guard db.isFinite else { return minDb }
        return min(0, max(minDb, db))
    }

    /// How many of the `segments` meter segments are lit for `db`.
    static func litSegments(_ db: Float) -> Int {
        Int((normalized(db) * Float(segments)).rounded())
    }

    /// 0-1 bar height for the waveform.
    static func normalized(_ db: Float) -> Float {
        (clamp(db) - minDb) / -minDb
    }

    /// `00:MM:SS`.
    static func formatTimer(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        return String(format: "%02d:%02d:%02d", total / 3600, (total / 60) % 60, total % 60)
    }
}
