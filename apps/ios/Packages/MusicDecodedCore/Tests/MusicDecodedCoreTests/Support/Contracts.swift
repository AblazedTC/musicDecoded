import Foundation
@testable import MusicDecodedCore

/// Loads the real files from the repo's `contracts/` folder (found by walking up from this
/// source file), so the tests always run against the shared contracts, never a copy.
enum Contracts {
    struct NotFound: Error, CustomStringConvertible {
        let path: String
        var description: String { "contracts/\(path) not found above \(#filePath)" }
    }

    static func url(_ path: String) throws -> URL {
        var directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<16 {
            let candidate = directory.appendingPathComponent("contracts").appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            directory = directory.deletingLastPathComponent()
        }
        throw NotFound(path: path)
    }

    static func sampleAnalysis() throws -> Analysis {
        try Analysis(data: Data(contentsOf: url("fixtures/sample.analysis.json")))
    }

    static func voicings() throws -> ChordVoicings {
        try ChordVoicings(contentsOf: url("chord-voicings.json"))
    }
}
