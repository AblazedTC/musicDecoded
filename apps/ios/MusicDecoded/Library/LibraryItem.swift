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

    /// Visible items, newest first. Use with `@Query(LibraryItem.newestFirst)` or `ModelContext.fetch`.
    static var newestFirst: FetchDescriptor<LibraryItem> {
        FetchDescriptor(
            predicate: #Predicate<LibraryItem> { item in
                !item.tombstoned
            },
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
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
}
