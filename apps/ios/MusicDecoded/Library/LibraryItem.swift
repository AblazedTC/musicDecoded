import Foundation
import MusicDecodedCore
import SwiftData

/// One track in this device's history. Mirrors `LibraryItem` in contracts/openapi.yaml plus the
/// local-only fields from docs/data-model.md.
@Model
final class LibraryItem {
    @Attribute(.unique) var trackId: String
    var title: String
    var artist: String?
    /// `Analysis.Source` raw value, stored as text so predicates can filter on it. Use `source`.
    var sourceRawValue: String
    /// YouTube videoId or SoundCloud URL.
    var sourceRef: String?
    var durationSec: Double?
    var thumbnailUrl: String?
    /// File/recording only: lets another device download the user's own audio.
    var audioUploadId: String?
    var favorite: Bool
    /// Display key copied from the analysis file ("B minor"), so lists can show and search it.
    var key: String?
    var bpm: Double?
    var lastPlayedAt: Date?
    var createdAt: Date
    /// Drives last-write-wins sync. Bump it on every user-visible change.
    var updatedAt: Date
    /// The contract's `deleted` tombstone. Renamed because `deleted`/`isDeleted` clash with
    /// Core Data and SwiftData members.
    var tombstoned: Bool

    // Local only
    /// `Status` raw value. Use `status`.
    var statusRawValue: String
    /// The analysis job while `status` is `.analyzing`.
    var jobId: String?
    /// Audio file name inside Application Support/audio, or `nil` for streaming-only tracks.
    /// Relative, because the container path changes between installs.
    var localAudioPath: String?
    var syncedAt: Date?
    /// The last failure, shown with Retry while `status` is `.failed`.
    var error: String?

    init(
        trackId: String,
        title: String,
        artist: String? = nil,
        source: Analysis.Source,
        sourceRef: String? = nil,
        durationSec: Double? = nil,
        thumbnailUrl: String? = nil,
        status: Status = .analyzing,
        jobId: String? = nil,
        createdAt: Date = .now
    ) {
        self.trackId = trackId
        self.title = title
        self.artist = artist
        self.sourceRawValue = source.rawValue
        self.sourceRef = sourceRef
        self.durationSec = durationSec
        self.thumbnailUrl = thumbnailUrl
        self.audioUploadId = nil
        self.favorite = false
        self.key = nil
        self.bpm = nil
        self.lastPlayedAt = nil
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.tombstoned = false
        self.statusRawValue = status.rawValue
        self.jobId = jobId
        self.localAudioPath = nil
        self.syncedAt = nil
        self.error = nil
    }
}

extension LibraryItem {
    /// Local analysis state.
    enum Status: String, Codable, Sendable {
        case analyzing, ready, failed
    }

    var source: Analysis.Source {
        get { Analysis.Source(rawValue: sourceRawValue) ?? .file }
        set { sourceRawValue = newValue.rawValue }
    }

    var status: Status {
        get { Status(rawValue: statusRawValue) ?? .failed }
        set { statusRawValue = newValue.rawValue }
    }

    /// Deletes are tombstones so they can sync: the row stays, hidden from lists.
    func markDeleted(at date: Date = .now) {
        tombstoned = true
        updatedAt = date
    }

    /// Favorites sync too, so this bumps `updatedAt`.
    func setFavorite(_ favorite: Bool, at date: Date = .now) {
        self.favorite = favorite
        updatedAt = date
    }

    /// The `limit` newest visible items (Home "Recently Decoded").
    static func recent(limit: Int) -> FetchDescriptor<LibraryItem> {
        var descriptor = newestFirst
        descriptor.fetchLimit = limit
        return descriptor
    }

    /// Visible items, newest first. Use with `@Query(LibraryItem.newestFirst)` or `ModelContext.fetch`.
    static var newestFirst: FetchDescriptor<LibraryItem> {
        FetchDescriptor(
            predicate: #Predicate<LibraryItem> { item in
                !item.tombstoned
            },
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
    }

    /// Library search (DESIGN.md §3.2), newest first: `query.text` against title, artist and key
    /// (case- and diacritic-insensitive), narrowed by the filter. Runs in the store.
    static func search(_ query: LibraryQuery) -> FetchDescriptor<LibraryItem> {
        let text = query.text
        let anyText = text.isEmpty
        let predicate: Predicate<LibraryItem>
        switch query.filter {
        case .all:
            predicate = #Predicate<LibraryItem> { item in
                !item.tombstoned
                    && (anyText
                        || item.title.localizedStandardContains(text)
                        || (item.artist?.localizedStandardContains(text) ?? false)
                        || (item.key?.localizedStandardContains(text) ?? false))
            }
        case .favorites:
            predicate = #Predicate<LibraryItem> { item in
                !item.tombstoned && item.favorite
                    && (anyText
                        || item.title.localizedStandardContains(text)
                        || (item.artist?.localizedStandardContains(text) ?? false)
                        || (item.key?.localizedStandardContains(text) ?? false))
            }
        case .offline:
            predicate = #Predicate<LibraryItem> { item in
                !item.tombstoned && item.localAudioPath != nil
                    && (anyText
                        || item.title.localizedStandardContains(text)
                        || (item.artist?.localizedStandardContains(text) ?? false)
                        || (item.key?.localizedStandardContains(text) ?? false))
            }
        case .recordings:
            let recording = Analysis.Source.recording.rawValue
            predicate = #Predicate<LibraryItem> { item in
                !item.tombstoned && item.sourceRawValue == recording
                    && (anyText
                        || item.title.localizedStandardContains(text)
                        || (item.artist?.localizedStandardContains(text) ?? false)
                        || (item.key?.localizedStandardContains(text) ?? false))
            }
        }
        return FetchDescriptor(
            predicate: predicate,
            sortBy: [SortDescriptor(\.createdAt, order: .reverse), SortDescriptor(\.trackId)]
        )
    }

    /// Visible items whose analysis is ready (the Library's "songs decoded" count).
    static var decodedItems: FetchDescriptor<LibraryItem> {
        let ready = Status.ready.rawValue
        return FetchDescriptor(
            predicate: #Predicate<LibraryItem> { item in
                !item.tombstoned && item.statusRawValue == ready
            }
        )
    }

    /// Visible items with audio on this device (the "available offline" count).
    static var offlineItems: FetchDescriptor<LibraryItem> {
        FetchDescriptor(
            predicate: #Predicate<LibraryItem> { item in
                !item.tombstoned && item.localAudioPath != nil
            }
        )
    }

    /// The row for `trackId`, including tombstones.
    static func withTrackId(_ trackId: String) -> FetchDescriptor<LibraryItem> {
        var descriptor = FetchDescriptor(
            predicate: #Predicate<LibraryItem> { item in
                item.trackId == trackId
            }
        )
        descriptor.fetchLimit = 1
        return descriptor
    }

    /// Visible items whose analysis is still running (polling resumes for these on launch).
    static var analyzingItems: FetchDescriptor<LibraryItem> {
        let analyzing = Status.analyzing.rawValue
        return FetchDescriptor(
            predicate: #Predicate<LibraryItem> { item in
                item.statusRawValue == analyzing && !item.tombstoned
            }
        )
    }

    /// Copies the display fields from a finished analysis and marks the item ready.
    func apply(_ analysis: Analysis, at date: Date = .now) {
        durationSec = analysis.meta.durationSec
        key = analysis.key.map { "\($0.tonic) \($0.mode.rawValue)" }
        bpm = analysis.bpm
        status = .ready
        jobId = nil
        error = nil
        updatedAt = date
    }
}
