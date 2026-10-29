import Foundation

/// What one song row shows (Home and Library), from the library row and its running job.
struct SongRowState: Equatable, Sendable {
    let trackId: String
    let title: String
    let artist: String?
    /// `duration · key · BPM`.
    let meta: String
    let artworkURL: URL?
    let status: SongRowStatus
    /// The audio is on this device.
    let isOffline: Bool
    let favorite: Bool

    init(_ item: LibraryItem, job: Job?) {
        trackId = item.trackId
        title = item.title
        artist = item.artist
        meta = SongMetadata.line(durationSec: item.durationSec, key: item.key, bpm: item.bpm)
        artworkURL = item.thumbnailUrl.flatMap(URL.init(string:))
        switch item.status {
        case .ready:
            status = .ready
        case .analyzing:
            status = .analyzing(progress: job.map { min(max($0.progress, 0), 100) })
        case .failed:
            status = .failed
        }
        isOffline = item.localAudioPath != nil
        favorite = item.favorite
    }
}
