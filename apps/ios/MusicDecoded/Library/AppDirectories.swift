import Foundation

/// The on-device file layout (PLAN.md §5.1):
/// `<Application Support>/audio/{trackId}.{ext}` and `<Application Support>/analysis/{trackId}.json`.
enum AppDirectories {
    static func applicationSupport() throws -> URL {
        try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
    }

    /// `analysis/`, created if needed. `base` defaults to Application Support.
    static func analysis(in base: URL? = nil) throws -> URL {
        try directory("analysis", in: base)
    }

    /// `audio/`, created if needed and excluded from iCloud backup (it can be large, and
    /// synced or re-downloadable audio shouldn't count against the user's backup).
    static func audio(in base: URL? = nil) throws -> URL {
        var url = try directory("audio", in: base)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
        return url
    }

    private static func directory(_ name: String, in base: URL?) throws -> URL {
        let parent = try base ?? applicationSupport()
        let url = parent.appending(path: name, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
