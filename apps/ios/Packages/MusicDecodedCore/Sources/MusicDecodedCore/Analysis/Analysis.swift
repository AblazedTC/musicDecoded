import Foundation

/// The chord file (schema v1). Mirrors `contracts/analysis.schema.json` field for field.
/// All times are seconds from the start of the audio.
public struct Analysis: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var meta: Meta
    /// Required by the schema, but may be `null`.
    public var key: Key?
    /// Required by the schema, but may be `null`.
    public var bpm: Double?
    /// Required by the schema, but may be `null`.
    public var timeSignature: TimeSignature?
    /// Sorted by `t` ascending.
    public var beats: [Beat]
    /// Sorted, contiguous, non-overlapping segments.
    public var chords: [Chord]
    public var spectrogram: Spectrogram?
    public var waveformPeaks: WaveformPeaks?

    public struct Meta: Codable, Sendable, Equatable {
        public var trackId: String
        public var source: Source
        /// YouTube videoId or SoundCloud URL; `nil` for file/recording.
        public var sourceRef: String?
        public var durationSec: Double
        /// ISO 8601 date-time, kept as text so the file round-trips verbatim.
        public var analyzedAt: String
        public var analyzerVersion: String
    }

    public enum Source: String, Codable, Sendable, CaseIterable {
        case file, recording, youtube, soundcloud
    }

    public struct Key: Codable, Sendable, Equatable {
        /// One of `C C# D D# E F F# G G# A A# B`.
        public var tonic: String
        public var mode: Mode
        public var confidence: Double?
    }

    public enum Mode: String, Codable, Sendable, CaseIterable {
        case major, minor
    }

    public struct TimeSignature: Codable, Sendable, Equatable {
        public var numerator: Int
        /// One of 2, 4, 8, 16.
        public var denominator: Int
    }

    public struct Beat: Codable, Sendable, Equatable {
        public var t: Double
        public var downbeat: Bool
    }

    public struct Chord: Codable, Sendable, Equatable {
        public var start: Double
        public var end: Double
        /// Harte-style label, e.g. `A:maj`, `F#:min7`, `D:maj/3`; `N` = no chord, `X` = unknown.
        public var label: String
        public var confidence: Double?
    }

    /// Downsampled mel spectrogram. Row-major frames: `data[frame * bins + bin]`.
    public struct Spectrogram: Codable, Sendable, Equatable {
        public var fps: Double
        public var bins: Int
        public var frames: Int
        /// Always `uint8-base64`: each value 0-255 maps to dB in [-80, 0].
        public var encoding: String
        /// Base64 text.
        public var data: String
    }

    public struct WaveformPeaks: Codable, Sendable, Equatable {
        public var fps: Double
        /// Values in [0, 1].
        public var peaks: [Double]
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, meta, key, bpm, timeSignature, beats, chords, spectrogram, waveformPeaks
    }
}

extension Analysis {
    /// Decodes an analysis file, e.g. the contents of `analysis/{trackId}.json`.
    public init(data: Data) throws {
        self = try JSONDecoder().decode(Analysis.self, from: data)
    }

    /// Writes `key`, `bpm` and `timeSignature` as explicit `null` when missing,
    /// because the schema requires those keys to be present.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(meta, forKey: .meta)
        try container.encode(key, forKey: .key)
        try container.encode(bpm, forKey: .bpm)
        try container.encode(timeSignature, forKey: .timeSignature)
        try container.encode(beats, forKey: .beats)
        try container.encode(chords, forKey: .chords)
        try container.encodeIfPresent(spectrogram, forKey: .spectrogram)
        try container.encodeIfPresent(waveformPeaks, forKey: .waveformPeaks)
    }
}
