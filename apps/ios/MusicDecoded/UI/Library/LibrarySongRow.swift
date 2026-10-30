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
    @Environment(NowPlaying.self) private var nowPlaying
    @Environment(\.openPaywall) private var openPaywall
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
        // One VoiceOver stop per song; the ⋯ menu and Retry become its actions.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(row.accessibilityLabel)
        .accessibilityAddTraits(row.status == .failed ? [] : .isButton)
        .accessibilityAction {
            open(row)
        }
        .accessibilityActions {
            if row.status == .failed {
                Button("Retry analysis") {
                    reanalyze()
                }
            }
            Button(row.favorite ? "Remove from favorites" : "Favorite") {
                setFavorite(!row.favorite)
            }
            if item.status != .analyzing {
                Button("Re-analyze") {
                    reanalyze()
                }
            }
            Button("Delete") {
                confirmDelete = true
            }
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
        LibraryActions.setFavorite(item, favorite, context: modelContext)
    }

    private func delete() {
        LibraryActions.delete(item, nowPlaying: nowPlaying, context: modelContext)
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
                let message = Transcription.userMessage(for: error)
                // The daily plan limit (a re-analyze the user just asked for) opens the paywall.
                if let openPaywall, OpenPaywallAction.opens(for: message) {
                    openPaywall()
                } else {
                    errorMessage = message
                    showError = true
                }
            }
        }
    }
}
