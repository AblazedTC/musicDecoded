import Foundation
import MusicDecodedCore

/// Analyses for the player tests. The core types' memberwise initializers are internal to
/// MusicDecodedCore, so these are decoded from JSON like a real analysis file.
enum PlayerFixtures {
    private final class BundleToken {}

    /// A small hand-made analysis with runs of `N` (same as the core tests' `SyntheticAnalysis`):
    ///
    ///     chords: A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10
    ///     beats:  every 1 s from 0 to 9, downbeat every 4 (0, 4, 8)
    static func synthetic(source: Analysis.Source = .file, sourceRef: String? = nil, beats: Bool = true, durationSec: Double = 10) throws -> Analysis {
        let beatList = beats ? (0..<10).map { #"{"t": \#($0), "downbeat": \#($0 % 4 == 0)}"# } : []
        let ref = sourceRef.map { "\"\($0)\"" } ?? "null"
        let json = """
        {
          "schemaVersion": 1,
          "meta": {"trackId": "synthetic", "source": "\(source.rawValue)", "sourceRef": \(ref),
                   "durationSec": \(durationSec), "analyzedAt": "2026-01-01T00:00:00Z", "analyzerVersion": "test"},
          "key": null,
          "bpm": 60,
          "timeSignature": {"numerator": 4, "denominator": 4},
          "beats": [\(beatList.joined(separator: ","))],
          "chords": [
            {"start": 0, "end": 2, "label": "A:maj"},
            {"start": 2, "end": 4, "label": "N"},
            {"start": 4, "end": 5, "label": "N"},
            {"start": 5, "end": 8, "label": "D:maj"},
            {"start": 8, "end": 10, "label": "N"}
          ],
          "spectrogram": null,
          "waveformPeaks": null
        }
        """
        return try Analysis(data: Data(json.utf8))
    }

    /// `contracts/fixtures/sunnydays.analysis.json`, added to the test bundle by reference.
    static func sunnyDays() throws -> Analysis {
        let bundle = Bundle(for: BundleToken.self)
        guard let url = bundle.url(forResource: "sunnydays.analysis", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try Analysis(data: Data(contentsOf: url))
    }

    /// `contracts/chord-voicings.json`, added to the test bundle by reference.
    static func voicings() throws -> ChordVoicings {
        let bundle = Bundle(for: BundleToken.self)
        guard let url = bundle.url(forResource: "chord-voicings", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try ChordVoicings(contentsOf: url)
    }
}
