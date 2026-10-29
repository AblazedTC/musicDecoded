import Foundation

/// The Library's filter chips (DESIGN.md §3.2).
enum LibraryFilter: String, CaseIterable, Hashable, Sendable {
    case all, favorites, offline, recordings

    var title: String {
        switch self {
        case .all: "All songs"
        case .favorites: "Favorites"
        case .offline: "Offline"
        case .recordings: "Recordings"
        }
    }
}

/// What the Library fetches for a search text + filter (see `LibraryItem.search`).
struct LibraryQuery: Hashable, Sendable {
    /// Trimmed, with ♯/♭ mapped to the `#`/`b` stored in keys. Empty = match everything.
    let text: String
    let filter: LibraryFilter

    init(text: String, filter: LibraryFilter) {
        self.text = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "♯", with: "#")
            .replacingOccurrences(of: "♭", with: "b")
        self.filter = filter
    }

    /// No text and no filter: an empty result means the library itself is empty.
    var isUnfiltered: Bool {
        text.isEmpty && filter == .all
    }
}
