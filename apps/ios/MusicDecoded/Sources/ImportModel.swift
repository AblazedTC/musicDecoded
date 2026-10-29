import Foundation
import Observation

/// Files / Media Library import (T-205i): pick → copy → Import sheet → `TranscriptionService.start`.
@MainActor
@Observable
final class ImportModel {
    /// Copying or exporting the picked audio.
    private(set) var isPreparing = false
    private(set) var draft: ImportDraft?
    /// Upload + job request in progress.
    private(set) var isStarting = false
    var errorMessage: String?

    /// Bumped by `reset()`, so a copy that finishes after the sheet closed is thrown away.
    @ObservationIgnored private var generation = 0

    init(draft: ImportDraft? = nil) {
        self.draft = draft
    }

    /// Whether the Import sheet has anything to show.
    var hasContent: Bool {
        isPreparing || draft != nil || errorMessage != nil
    }

    func importFile(_ url: URL, using service: TranscriptionService) async {
        await prepare {
            guard let fileExtension = AudioImporter.supportedExtension(for: url) else {
                throw TranscriptionError(message: Transcription.unsupportedMessage)
            }
            let pendingURL = try service.newPendingFileURL(extension: fileExtension)
            return try await AudioImporter.importFile(at: url, to: pendingURL)
        }
    }

    func importMediaItem(_ pick: MediaLibraryPick, using service: TranscriptionService) async {
        await prepare {
            let pendingURL = try service.newPendingFileURL(extension: "m4a")
            return try await AudioImporter.exportMediaItem(pick, to: pendingURL)
        }
    }

    /// Uploads the draft and starts the job. Returns the trackId, or `nil` on failure (the error is
    /// shown in the sheet and the draft is kept for another try).
    func importAndDecode(using service: TranscriptionService) async -> String? {
        guard let draft, !isStarting else { return nil }
        isStarting = true
        errorMessage = nil
        defer { isStarting = false }
        do {
            let trackId = try await service.start(
                LocalAudio(fileURL: draft.fileURL, title: draft.title, source: .file, durationSec: draft.durationSec)
            )
            self.draft = nil
            return trackId
        } catch {
            errorMessage = Transcription.userMessage(for: error)
            return nil
        }
    }

    /// Deletes the pending copy and clears the sheet. Does nothing while an upload runs.
    func reset() {
        guard !isStarting else { return }
        generation += 1
        if let draft {
            try? FileManager.default.removeItem(at: draft.fileURL)
        }
        draft = nil
        errorMessage = nil
    }

    private func prepare(_ work: () async throws -> ImportDraft) async {
        reset()
        let current = generation
        isPreparing = true
        defer { isPreparing = false }
        do {
            let prepared = try await work()
            guard generation == current else {
                // The sheet was closed while copying: nobody wants this copy any more.
                try? FileManager.default.removeItem(at: prepared.fileURL)
                return
            }
            draft = prepared
        } catch {
            if generation == current {
                errorMessage = Transcription.userMessage(for: error)
            }
        }
    }
}
