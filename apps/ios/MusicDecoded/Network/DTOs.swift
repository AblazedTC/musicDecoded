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
