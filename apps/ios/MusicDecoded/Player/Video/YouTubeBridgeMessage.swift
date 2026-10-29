import Foundation

/// `YT.PlayerState` from the IFrame Player API.
enum YouTubePlayerState: Int, Sendable {
    case unstarted = -1
    case ended = 0
    case playing = 1
    case paused = 2
    case buffering = 3
    case cued = 5
}

/// A message the embed page posts through `window.webkit.messageHandlers.ytBridge`
/// (see `YouTubePlayerPage`). Parsed off the raw `WKScriptMessage.body`.
enum YouTubeBridgeMessage: Equatable, Sendable {
    /// The player is ready for commands.
    case ready
    /// `onStateChange`, with the time when it happened.
    case state(YouTubePlayerState, time: Double?)
    /// The periodic `getCurrentTime()` report.
    case time(Double)
    /// `onPlaybackRateChange`.
    case rate(Double)
    /// `onError` (2, 5, 100, 101, 150), or -1 when the API script couldn't load (offline).
    case error(Int)

    /// `nil` for anything that isn't a known message.
    init?(body: Any) {
        guard let fields = body as? [String: Any], let event = fields["event"] as? String else { return nil }
        switch event {
        case "ready":
            self = .ready
        case "state":
            guard
                let raw = Self.number(fields["state"]),
                let state = YouTubePlayerState(rawValue: Int(raw))
            else { return nil }
            self = .state(state, time: Self.number(fields["time"]))
        case "time":
            guard let time = Self.number(fields["time"]) else { return nil }
            self = .time(time)
        case "rate":
            guard let rate = Self.number(fields["rate"]), rate > 0 else { return nil }
            self = .rate(rate)
        case "error":
            self = .error(Self.number(fields["code"]).map { Int($0) } ?? -1)
        default:
            return nil
        }
    }

    /// JS numbers arrive as `NSNumber`; tests pass Swift `Double`/`Int`.
    private static func number(_ value: Any?) -> Double? {
        guard let value else { return nil }
        let result: Double?
        if let double = value as? Double {
            result = double
        } else if let int = value as? Int {
            result = Double(int)
        } else {
            result = nil
        }
        guard let result, result.isFinite else { return nil }
        return result
    }
}
