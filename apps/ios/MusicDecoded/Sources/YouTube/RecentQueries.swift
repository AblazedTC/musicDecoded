import Foundation

/// The YouTube suggestion chips (DESIGN.md §3.4): the last queries searched on this device, newest
/// first, at most 8. Local only, so they cost no quota (PLAN.md R-6). Not secret: UserDefaults.
enum RecentQueries {
    static let limit = 8
    static let defaultsKey = "youtubeRecentQueries"

    /// `list` with `query` moved (or added) to the front, without case-insensitive duplicates.
    static func adding(_ query: String, to list: [String]) -> [String] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else { return list }
        var result = list.filter { $0.caseInsensitiveCompare(trimmed) != .orderedSame }
        result.insert(trimmed, at: 0)
        return Array(result.prefix(limit))
    }

    static func load(from defaults: UserDefaults) -> [String] {
        Array((defaults.stringArray(forKey: defaultsKey) ?? []).prefix(limit))
    }

    static func save(_ list: [String], to defaults: UserDefaults) {
        defaults.set(list, forKey: defaultsKey)
    }
}
