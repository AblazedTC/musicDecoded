import Foundation

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
