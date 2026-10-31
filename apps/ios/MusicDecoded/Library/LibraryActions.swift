import SwiftData

/// Item actions shared by song rows and the Player's ⋯ menu.
@MainActor
enum LibraryActions {
    /// Delete is a tombstone (it syncs): the row disappears from every list, and if the track is
    /// the one loaded, playback stops and the mini player goes away.
    static func delete(_ item: LibraryItem, nowPlaying: NowPlaying, context: ModelContext) {
        nowPlaying.stop(trackId: item.trackId)
        item.markDeleted()
        try? context.save()
    }

    static func setFavorite(_ item: LibraryItem, _ favorite: Bool, context: ModelContext) {
        item.setFavorite(favorite)
        try? context.save()
    }

    /// Playback went from stopped to playing.
    static func markPlayed(_ item: LibraryItem, context: ModelContext) {
        item.markPlayed()
        try? context.save()
    }
}
