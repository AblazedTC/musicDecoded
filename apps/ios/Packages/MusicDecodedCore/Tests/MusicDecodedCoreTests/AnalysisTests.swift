import Foundation
import Testing
@testable import MusicDecodedCore

struct AnalysisTests {
    @Test func decodesSampleFixture() throws {
        let analysis = try Contracts.sampleAnalysis()

        #expect(analysis.schemaVersion == 1)
        #expect(analysis.meta.trackId == "fixture-bad")
        #expect(analysis.meta.source == .file)
        #expect(analysis.meta.sourceRef == nil)
        #expect(analysis.meta.durationSec == 263.407)
        #expect(analysis.key == Analysis.Key(tonic: "D", mode: .major, confidence: 0))
        #expect(analysis.bpm == 136.1)
        #expect(analysis.timeSignature == Analysis.TimeSignature(numerator: 4, denominator: 4))
        #expect(analysis.beats.count == 581)
        #expect(analysis.chords.count == 94)
        #expect(analysis.chords.first?.label == "N")
        #expect(analysis.chords[1] == Analysis.Chord(start: 0.72, end: 1.138, label: "A:maj"))
        #expect(analysis.spectrogram == nil)
        #expect(analysis.waveformPeaks == nil)
    }

    @Test func fixtureChordsAreSortedAndContiguous() throws {
        let chords = try Contracts.sampleAnalysis().chords
        for (previous, next) in zip(chords, chords.dropFirst()) {
            #expect(previous.end == next.start, "gap or overlap at \(previous.end)")
        }
    }

    @Test func decodesNullableFieldsAndVisuals() throws {
        let json = """
        {
          "schemaVersion": 1,
          "meta": { "trackId": "t1", "source": "youtube", "sourceRef": "abc123",
                    "durationSec": 3, "analyzedAt": "2026-01-01T00:00:00Z", "analyzerVersion": "1.0.0" },
          "key": null, "bpm": null, "timeSignature": null,
          "beats": [{ "t": 0.5, "downbeat": true }],
          "chords": [{ "start": 0, "end": 3, "label": "C:maj", "confidence": 0.9 }],
          "spectrogram": { "fps": 20, "bins": 2, "frames": 1, "encoding": "uint8-base64", "data": "AP8=" },
          "waveformPeaks": { "fps": 20, "peaks": [0, 0.5, 1] }
        }
        """
        let analysis = try Analysis(data: Data(json.utf8))

        #expect(analysis.meta.source == .youtube)
        #expect(analysis.meta.sourceRef == "abc123")
        #expect(analysis.key == nil)
        #expect(analysis.bpm == nil)
        #expect(analysis.timeSignature == nil)
        #expect(analysis.chords.first?.confidence == 0.9)
        #expect(analysis.spectrogram?.bins == 2)
        #expect(analysis.waveformPeaks?.peaks == [0, 0.5, 1])
    }

    @Test func roundTripsThroughJSON() throws {
        let original = try Contracts.sampleAnalysis()
        let data = try JSONEncoder().encode(original)
        #expect(try Analysis(data: data) == original)
    }

    @Test func encodesRequiredNullableKeysAsNull() throws {
        var analysis = SyntheticAnalysis.make()
        analysis.bpm = nil
        analysis.timeSignature = nil

        let data = try JSONEncoder().encode(analysis)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])

        #expect(object["key"] is NSNull)
        #expect(object["bpm"] is NSNull)
        #expect(object["timeSignature"] is NSNull)
        #expect(object.keys.contains("spectrogram") == false, "optional visuals are omitted, not null")
    }
}
