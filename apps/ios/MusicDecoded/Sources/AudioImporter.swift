import AVFoundation
import Foundation
import UniformTypeIdentifiers

/// A picked file, copied into app storage and waiting for "Import & decode" (DESIGN.md §3.3).
struct ImportDraft: Sendable, Equatable {
    /// Pending copy in `audio/`; renamed to `audio/{trackId}.{ext}` once the job exists.
    var fileURL: URL
    var fileName: String
    /// From the file's metadata, falling back to the file name.
    var title: String
    var sizeBytes: Int
    var durationSec: Double?
    /// Bits per second, if the file says.
    var bitrate: Int?

    var typeLabel: String { fileURL.pathExtension.uppercased() }
}

/// A song picked in the Media Library. Only songs with an `assetURL` (local, not DRM) get here.
struct MediaLibraryPick: Sendable {
    var assetURL: URL
    var title: String?
}

/// Copies picked audio into app storage and reads its metadata. Not actor-isolated: the copying,
/// exporting and metadata loading run off the main actor.
enum AudioImporter {
    static let unreadableMessage = "Couldn't read this file. Try another one."
    static let protectedMessage =
        "This song can't be imported. It may be protected or not downloaded to this device."

    /// An extension `POST /uploads` accepts, from the file name or else its type.
    static func supportedExtension(for url: URL) -> String? {
        let fromName = url.pathExtension.lowercased()
        if Transcription.contentType(forExtension: fromName) != nil {
            return fromName
        }
        let type = (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType
        guard let fromType = type?.preferredFilenameExtension?.lowercased() else { return nil }
        return Transcription.contentType(forExtension: fromType) == nil ? nil : fromType
    }

    /// Copies a file from `.fileImporter` (a security-scoped URL) to `pendingURL`.
    static func importFile(at url: URL, to pendingURL: URL) async throws -> ImportDraft {
        let accessing = url.startAccessingSecurityScopedResource()
        defer {
            if accessing { url.stopAccessingSecurityScopedResource() }
        }
        do {
            try FileManager.default.copyItem(at: url, to: pendingURL)
        } catch {
            try? FileManager.default.removeItem(at: pendingURL)
            throw TranscriptionError(message: unreadableMessage)
        }
        return await draft(for: pendingURL, fileName: url.lastPathComponent, fallbackTitle: nil)
    }

    /// Exports a Media Library song to an `.m4a` at `pendingURL`.
    static func exportMediaItem(_ pick: MediaLibraryPick, to pendingURL: URL) async throws -> ImportDraft {
        let asset = AVURLAsset(url: pick.assetURL)
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else {
            throw TranscriptionError(message: protectedMessage)
        }
        if #available(iOS 18, *) {
            do {
                try await session.export(to: pendingURL, as: .m4a)
            } catch {
                try? FileManager.default.removeItem(at: pendingURL)
                throw TranscriptionError(message: protectedMessage)
            }
        } else {
            session.outputURL = pendingURL
            session.outputFileType = .m4a
            await session.export()
            guard session.status == .completed else {
                try? FileManager.default.removeItem(at: pendingURL)
                throw TranscriptionError(message: protectedMessage)
            }
        }
        let name = pick.title.flatMap { $0.isEmpty ? nil : $0 } ?? "Audio"
        return await draft(for: pendingURL, fileName: "\(name).m4a", fallbackTitle: pick.title)
    }

    /// Reads title, duration and bitrate. Unreadable metadata isn't an error: the server decides.
    static func draft(for fileURL: URL, fileName: String, fallbackTitle: String?) async -> ImportDraft {
        let asset = AVURLAsset(url: fileURL)

        var durationSec: Double?
        if let duration = try? await asset.load(.duration) {
            let seconds = CMTimeGetSeconds(duration)
            if seconds.isFinite, seconds > 0 {
                durationSec = seconds
            }
        }

        var metadataTitle: String?
        if let metadata = try? await asset.load(.commonMetadata),
           let titleItem = AVMetadataItem.metadataItems(from: metadata, filteredByIdentifier: .commonIdentifierTitle).first,
           let value = try? await titleItem.load(.stringValue) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            metadataTitle = trimmed.isEmpty ? nil : trimmed
        }

        var bitrate: Int?
        if let tracks = try? await asset.loadTracks(withMediaType: .audio),
           let track = tracks.first,
           let rate = try? await track.load(.estimatedDataRate),
           rate > 0 {
            bitrate = Int(rate)
        }

        let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let nameTitle = (fileName as NSString).deletingPathExtension
        return ImportDraft(
            fileURL: fileURL,
            fileName: fileName,
            title: metadataTitle ?? fallbackTitle ?? nameTitle,
            sizeBytes: size,
            durationSec: durationSec,
            bitrate: bitrate
        )
    }
}
