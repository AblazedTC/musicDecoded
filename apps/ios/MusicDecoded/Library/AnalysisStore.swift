import Foundation
import MusicDecodedCore

/// Analysis files on disk, one per track: `analysis/{trackId}.json`, saved verbatim from the
/// server. The player only needs this file to work offline.
struct AnalysisStore: Sendable {
    struct InvalidTrackID: Error {
        let trackId: String
    }

    let directory: URL

    init(directory: URL) {
        self.directory = directory
    }

    /// The app's store in Application Support/analysis.
    static func appDefault() throws -> AnalysisStore {
        try AnalysisStore(directory: AppDirectories.analysis())
    }

    func fileURL(for trackId: String) throws -> URL {
        // trackIds come from the server; refuse anything that could leave the directory.
        guard !trackId.isEmpty, !trackId.contains("/"), !trackId.hasPrefix(".") else {
            throw InvalidTrackID(trackId: trackId)
        }
        return directory.appending(path: "\(trackId).json", directoryHint: .notDirectory)
    }

    func contains(trackId: String) -> Bool {
        guard let url = try? fileURL(for: trackId) else { return false }
        return FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
    }

    func load(trackId: String) throws -> Analysis {
        try Analysis(data: Data(contentsOf: fileURL(for: trackId)))
    }

    /// Checks that `data` decodes as an analysis file, then writes it unchanged.
    @discardableResult
    func save(_ data: Data, for trackId: String) throws -> Analysis {
        let analysis = try Analysis(data: data)
        let url = try fileURL(for: trackId)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        return analysis
    }

    func delete(trackId: String) throws {
        let url = try fileURL(for: trackId)
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else { return }
        try FileManager.default.removeItem(at: url)
    }
}
