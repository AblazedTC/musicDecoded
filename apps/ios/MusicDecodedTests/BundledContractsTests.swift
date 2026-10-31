import Foundation
import Testing
import MusicDecodedCore

/// The contracts files are added to this bundle by reference (see project.yml), so these
/// tests prove the app-side wiring: resources resolve and decode with MusicDecodedCore.
struct BundledContractsTests {
    private final class BundleToken {}

    private let bundle = Bundle(for: BundleToken.self)

    @Test func sampleAnalysisDecodes() throws {
        let url = try #require(bundle.url(forResource: "sample.analysis", withExtension: "json"))
        let analysis = try Analysis(data: Data(contentsOf: url))

        #expect(analysis.chords.isEmpty == false)
        #expect(currentState(at: 1, in: analysis).chordIndex != nil)
    }

    @Test func chordVoicingsMatchTheirExamples() throws {
        let url = try #require(bundle.url(forResource: "chord-voicings", withExtension: "json"))
        let voicings = try ChordVoicings(contentsOf: url)
        let examples = try #require(voicings.examples)

        for (label, expected) in examples {
            #expect(voicings.notes(for: label) == expected, "\(label)")
        }
    }
}
