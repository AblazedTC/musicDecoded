import SwiftData
import SwiftUI

/// A `SongRow` for a library item, with its ⋯ menu (Favorite, Re-analyze, Delete with
/// confirmation) and Retry on failed rows. Ready rows open the Player; analyzing rows open the
/// Processing screen. Shared by Home and Library.
struct LibrarySongRow: View {
    let item: LibraryItem
    let onOpenPlayer: (_ trackId: String) -> Void
    let onOpenProcessing: (_ trackId: String) -> Void

    @Environment(\.modelContext) private var modelContext
    @Environment(TranscriptionService.self) private var transcription
    @State private var confirmDelete = false
    @State private var errorMessage: String?
    @State private var showError = false

    var body: some View {
        let row = SongRowState(item, job: transcription.jobs[item.trackId])
        SongRow(
            title: row.title,
            artist: row.artist,
            metadata: row.meta,
            artworkURL: row.artworkURL,
            isOffline: row.isOffline,
            status: row.status,
            onRetry: { reanalyze() }
        ) {
            Button(
                row.favorite ? "Remove from favorites" : "Favorite",
                systemImage: row.favorite ? "heart.slash" : "heart"
            ) {
                setFavorite(!row.favorite)
            }
            Button("Re-analyze", systemImage: "arrow.clockwise") {
                reanalyze()
            }
            .disabled(item.status == .analyzing)
            Button("Delete", systemImage: "trash", role: .destructive) {
                confirmDelete = true
            }
        }
        .contentShape(.rect)
        .onTapGesture {
            open(row)
        }
        .accessibilityAction(named: "Open") {
            open(row)
        }
        .confirmationDialog("Delete “\(row.title)”?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                delete()
            }
        } message: {
            Text("It's removed from your library and its chord timeline won't show up here anymore.")
        }
        .alert("Couldn't re-analyze", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? Transcription.genericMessage)
        }
    }

    private func open(_ row: SongRowState) {
        switch row.status {
        case .ready:
            onOpenPlayer(row.trackId)
        case .analyzing:
            onOpenProcessing(row.trackId)
        case .failed:
            break
        }
    }

    private func setFavorite(_ favorite: Bool) {
        item.setFavorite(favorite)
        try? modelContext.save()
    }

    /// Delete is a tombstone (it syncs); the row disappears from every list.
    private func delete() {
        item.markDeleted()
        try? modelContext.save()
    }

    /// Retry on a failed row, or Re-analyze from the menu. A failure to start is shown once.
    private func reanalyze() {
        let trackId = item.trackId
        Task {
            do {
                try await transcription.retry(trackId: trackId)
            } catch is CancellationError {
                return
            } catch {
                errorMessage = Transcription.userMessage(for: error)
                showError = true
            }
        }
    }
}
