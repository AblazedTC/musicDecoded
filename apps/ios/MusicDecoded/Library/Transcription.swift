import CryptoKit
import Foundation
import MusicDecodedCore

/// Audio on this device, ready to be decoded (a picked file or a finished recording).
struct LocalAudio: Sendable {
    /// A file from `TranscriptionService.newPendingFileURL`; it's moved to `audio/{trackId}.{ext}`.
    var fileURL: URL
    var title: String
    /// `.file` or `.recording`.
    var source: Analysis.Source
    var durationSec: Double?
}

/// A failure whose message can be shown to the user as is.
struct TranscriptionError: LocalizedError, Equatable, Sendable {
    let message: String

    var errorDescription: String? { message }
}

/// Copy and pure helpers for the job runner. Not actor-isolated, so they're usable anywhere.
enum Transcription {
    static let maxUploadBytes = 104_857_600

    static let rateLimitedMessage =
        "You've reached today's decode limit on the free plan. Try again tomorrow, or upgrade to Pro for more."
    static let tooLargeMessage = "This file is too large to upload (the limit is 100 MB)."
    static let offlineMessage = "Couldn't reach MusicDecoded. Check your connection and try again."
    static let analysisFailedMessage = "We couldn't decode this audio. Try again."
    static let expiredMessage = "This analysis expired. Try again."
    static let genericMessage = "Something went wrong. Try again."
    static let unsupportedMessage = "This audio format isn't supported. Try MP3, M4A, AAC, WAV, FLAC or OGG."
    static let emptyMessage = "This file is empty."
    static let missingAudioMessage = "The audio for this song is no longer on this device."

    /// The upload content type `POST /uploads` allows for a file extension, or `nil`.
    static func contentType(forExtension fileExtension: String) -> String? {
        switch fileExtension.lowercased() {
        case "mp3": "audio/mpeg"
        case "m4a", "mp4": "audio/mp4"
        case "aac": "audio/aac"
        case "wav": "audio/wav"
        case "flac": "audio/flac"
        case "ogg", "oga", "opus": "audio/ogg"
        default: nil
        }
    }

    /// A message for any failure, with `rate_limited` (429) and `too_large` (413) spelled out.
    static func userMessage(for error: any Error) -> String {
        switch error {
        case let error as TranscriptionError:
            return error.message
        case let error as APIError:
            if error.code == .rateLimited || error.status == 429 { return rateLimitedMessage }
            if error.code == .tooLarge || error.status == 413 { return tooLargeMessage }
            return error.message.isEmpty ? genericMessage : error.message
        case is URLError:
            return offlineMessage
        default:
            return genericMessage
        }
    }

    /// Lowercase hex SHA-256 of the file, read in chunks. `nonisolated async`, so it runs off the
    /// main actor.
    static func sha256(of fileURL: URL) async throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 16), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// trackIds come from the server; only these are used as file names.
    static func isSafeTrackId(_ trackId: String) -> Bool {
        trackId.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil
    }
}
