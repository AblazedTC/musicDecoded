import Foundation
import MusicDecodedCore

// Request/response bodies from contracts/openapi.yaml. Type and field names match the
// schemas exactly; inline request bodies are named after their endpoint.

/// `Plan`: server-side entitlement (PLAN.md §9).
enum Plan: String, Codable, Sendable {
    case free, pro
}

/// Body of `POST /auth/anonymous`.
struct AnonymousAuthRequest: Encodable, Sendable {
    var deviceId: String
    var platform = "ios"
}

/// `AuthResponse`.
struct AuthResponse: Codable, Sendable, Equatable {
    enum Kind: String, Codable, Sendable {
        case guest, account
    }

    var token: String
    var userId: String
    var kind: Kind
    var plan: Plan
}

/// `ClientConfig`: server-controlled flags, read at launch.
struct ClientConfig: Codable, Sendable, Equatable {
    struct Limits: Codable, Sendable, Equatable {
        /// `nil` means unlimited.
        var dailyJobs: Int?
        var jobsUsedToday: Int
    }

    /// PLAN.md R-1. `false` in store builds.
    var offlineStreamAudio: Bool
    var maxUploadBytes: Int
    /// Optional in the contract; treat a missing value as `false`.
    var soundcloudEnabled: Bool?
    var plan: Plan
    var limits: Limits
}

/// Body of `POST /uploads`.
struct UploadRequest: Encodable, Sendable, Equatable {
    /// One of the content types `POST /uploads` allows (`audio/mp4`, `audio/mpeg`, ...).
    var contentType: String
    var sizeBytes: Int
    /// Lowercase hex SHA-256 of the bytes.
    var sha256: String
}

/// `201` body of `POST /uploads`: PUT the raw bytes to `putUrl` with the same Content-Type.
struct UploadTarget: Decodable, Sendable, Equatable {
    var uploadId: String
    var putUrl: URL
    var expiresAt: String
}

/// `JobRequest`. `uploadId` for file/recording, `videoId` for youtube, `soundcloudUrl` for soundcloud.
struct JobRequest: Encodable, Sendable, Equatable {
    var source: Analysis.Source
    var uploadId: String?
    var videoId: String?
    var soundcloudUrl: String?
    /// Display title for file/recording.
    var title: String?
}

/// `Job`.
struct Job: Decodable, Sendable, Equatable {
    enum Status: String, Decodable, Sendable {
        case queued, processing, complete, failed
    }

    var jobId: String
    var trackId: String
    var status: Status
    /// `fetching_media` ... `saving_results`, or `nil`. Kept as text so a new stage can't break polling.
    var stage: String?
    /// 0-100.
    var progress: Int
    var error: String?
    /// Track metadata once the worker knows it (e.g. yt-dlp for streaming sources).
    var title: String? = nil
    var thumbnailUrl: String? = nil
    var durationSec: Double? = nil
}

/// `VideoSummary`: one `GET /youtube/search` result.
struct VideoSummary: Codable, Sendable, Hashable, Identifiable {
    var videoId: String
    var title: String
    var channelTitle: String
    var thumbnailUrl: URL
    var durationSec: Int?

    var id: String { videoId }
}

/// `200` body of `GET /youtube/search`.
struct YouTubeSearchPage: Decodable, Sendable, Equatable {
    var nextPageToken: String?
    var items: [VideoSummary]
}

/// `200` body of `GET /tracks/{trackId}/analysis`.
struct AnalysisLocation: Decodable, Sendable, Equatable {
    var url: URL
    var schemaVersion: Int
}

/// `Credentials`: body of `POST /auth/signup` and `POST /auth/login`.
struct Credentials: Encodable, Sendable, Equatable {
    var email: String
    var password: String
}

/// `LibraryItem` in contracts/openapi.yaml, as sent over `/sync/library`. Times are ISO-8601
/// `date-time` strings (`SyncTime` converts them). Named `SyncItem` because `LibraryItem` is the
/// SwiftData row.
struct SyncItem: Codable, Sendable, Equatable {
    var trackId: String
    var title: String
    var artist: String?
    var source: Analysis.Source
    var sourceRef: String?
    var durationSec: Double?
    var thumbnailUrl: String?
    var audioUploadId: String?
    var favorite: Bool
    var lastPlayedAt: String?
    var createdAt: String
    var updatedAt: String
    /// The tombstone (`LibraryItem.tombstoned` on the device).
    var deleted: Bool

    init(
        trackId: String,
        title: String,
        artist: String? = nil,
        source: Analysis.Source,
        sourceRef: String? = nil,
        durationSec: Double? = nil,
        thumbnailUrl: String? = nil,
        audioUploadId: String? = nil,
        favorite: Bool = false,
        lastPlayedAt: String? = nil,
        createdAt: String,
        updatedAt: String,
        deleted: Bool
    ) {
        self.trackId = trackId
        self.title = title
        self.artist = artist
        self.source = source
        self.sourceRef = sourceRef
        self.durationSec = durationSec
        self.thumbnailUrl = thumbnailUrl
        self.audioUploadId = audioUploadId
        self.favorite = favorite
        self.lastPlayedAt = lastPlayedAt
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.deleted = deleted
    }

    private enum CodingKeys: String, CodingKey {
        case trackId, title, artist, source, sourceRef, durationSec, thumbnailUrl, audioUploadId
        case favorite, lastPlayedAt, createdAt, updatedAt, deleted
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        trackId = try container.decode(String.self, forKey: .trackId)
        title = try container.decode(String.self, forKey: .title)
        artist = try container.decodeIfPresent(String.self, forKey: .artist)
        source = try container.decode(Analysis.Source.self, forKey: .source)
        sourceRef = try container.decodeIfPresent(String.self, forKey: .sourceRef)
        durationSec = try container.decodeIfPresent(Double.self, forKey: .durationSec)
        thumbnailUrl = try container.decodeIfPresent(String.self, forKey: .thumbnailUrl)
        audioUploadId = try container.decodeIfPresent(String.self, forKey: .audioUploadId)
        // `favorite` defaults to false in the contract.
        favorite = try container.decodeIfPresent(Bool.self, forKey: .favorite) ?? false
        lastPlayedAt = try container.decodeIfPresent(String.self, forKey: .lastPlayedAt)
        createdAt = try container.decode(String.self, forKey: .createdAt)
        updatedAt = try container.decode(String.self, forKey: .updatedAt)
        deleted = try container.decode(Bool.self, forKey: .deleted)
    }
}

/// `200` body of `GET /sync/library`.
struct SyncPage: Codable, Sendable, Equatable {
    var items: [SyncItem]
    var cursor: String
}

/// Body of `PUT /sync/library`, and its `200` response (where `items` is optional).
struct SyncItems: Codable, Sendable, Equatable {
    var items: [SyncItem]

    init(items: [SyncItem]) {
        self.items = items
    }

    private enum CodingKeys: String, CodingKey {
        case items
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        items = try container.decodeIfPresent([SyncItem].self, forKey: .items) ?? []
    }
}

/// `200` body of `GET /uploads/{uploadId}`: a short-lived presigned GET for the caller's own audio.
struct UploadLocation: Decodable, Sendable, Equatable {
    var url: URL
    var contentType: String
    var expiresAt: String
}
