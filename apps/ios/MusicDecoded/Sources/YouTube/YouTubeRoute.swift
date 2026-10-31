import SwiftUI

/// The YouTube sandbox screens (DESIGN.md §2: pushed, no tab bar).
enum YouTubeRoute: Hashable, Sendable {
    case search
    case video(VideoSummary)
}

extension VideoSummary {
    /// A video known only by its id (a pasted link): the title and channel aren't known on the
    /// device, so they stay empty and the screens show a fallback.
    init(pastedVideoId videoId: String) {
        self.init(
            videoId: videoId,
            title: "",
            channelTitle: "",
            // YouTube's public thumbnail URL for any video.
            thumbnailUrl: URL(string: "https://i.ytimg.com/vi/\(videoId)/hqdefault.jpg")!,
            durationSec: nil
        )
    }

    var displayTitle: String {
        title.isEmpty ? Transcription.youtubeFallbackTitle : title
    }

    /// `m:ss` for the duration badge, or `nil` when it's unknown.
    var durationText: String? {
        durationSec.map { PlayerFormat.time(Double($0)) }
    }
}

extension View {
    /// Search and video detail, pushed onto the caller's `path`. `onDecodeStarted` gets the
    /// trackId once the job exists.
    func youtubeDestinations(
        path: Binding<NavigationPath>,
        onDecodeStarted: @escaping (_ trackId: String) -> Void
    ) -> some View {
        navigationDestination(for: YouTubeRoute.self) { route in
            switch route {
            case .search:
                YouTubeSearchScreen { video in
                    path.wrappedValue.append(YouTubeRoute.video(video))
                }
            case .video(let video):
                VideoDetailScreen(video: video, onDecodeStarted: onDecodeStarted)
            }
        }
    }
}
