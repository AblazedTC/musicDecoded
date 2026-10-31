import Foundation
import MusicDecodedCore

/// Pushes the Player for a track (DESIGN.md §2: pushed, no tab bar). Use with
/// `.navigationDestination(for: PlayerRoute.self) { PlayerScreen(trackId: $0.trackId) }`.
struct PlayerRoute: Hashable, Sendable {
    let trackId: String
}

/// Builds a `PlayerTrack` from the files on this device. Runs off the main actor.
enum PlayerTrackLoader {
    /// What the library row says about the track (read on the main actor first).
    struct Info: Sendable {
        var title: String
        var artist: String?
        var favorite: Bool
        var audioURL: URL?
        var artworkURL: URL? = nil
    }

    /// `analysis/{trackId}.json` (or, in Debug, a bundled `fixture-*` analysis) plus the local audio
    /// when its file exists. `nil` if there's no analysis on this device.
    static func load(trackId: String, info: Info) async -> PlayerTrack? {
        if let analysis = try? AnalysisStore.appDefault().load(trackId: trackId) {
            let audioURL = info.audioURL.flatMap { url in
                FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) ? url : nil
            }
            return PlayerTrack(
                analysis: analysis,
                title: info.title,
                artist: info.artist,
                favorite: info.favorite,
                audioURL: audioURL,
                artworkURL: info.artworkURL
            )
        }
        #if DEBUG
        if let fixture = DebugFixtures.load(trackId: trackId) {
            return PlayerTrack(
                analysis: fixture,
                title: trackId == DebugFixtures.sunnyDays ? "Sunny Days" : info.title,
                artist: info.artist ?? "Fixture",
                favorite: info.favorite,
                audioURL: nil,
                artworkURL: info.artworkURL
            )
        }
        #endif
        return nil
    }
}
