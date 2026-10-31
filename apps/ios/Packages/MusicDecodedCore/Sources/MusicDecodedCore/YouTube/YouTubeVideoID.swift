import Foundation

/// Pulls the videoId out of a pasted YouTube link (DESIGN.md §3.4, "Paste a video URL").
///
/// Understands `youtube.com/watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/` and
/// `music.youtube.com/watch?v=`, on `www.`/`m.` hosts too, with or without a scheme. Extra query
/// items (`si`, `t`, `list`, ...) are ignored. A bare id isn't accepted: plain text could look like one.
public enum YouTubeVideoID {
    /// The videoId in `text`, or `nil` if it isn't a YouTube video link.
    public static func parse(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else { return nil }
        let urlText = trimmed.contains("://") ? trimmed : "https://" + trimmed
        guard
            let components = URLComponents(string: urlText),
            let scheme = components.scheme?.lowercased(),
            scheme == "https" || scheme == "http",
            var host = components.host?.lowercased()
        else { return nil }
        for prefix in ["www.", "m."] where host.hasPrefix(prefix) {
            host.removeFirst(prefix.count)
        }

        let segments = components.path.split(separator: "/").map(String.init)
        let candidate: String?
        switch host {
        case "youtu.be":
            candidate = segments.first
        case "youtube.com", "music.youtube.com", "youtube-nocookie.com":
            if segments == ["watch"] {
                candidate = components.queryItems?.first { $0.name == "v" }?.value
            } else if segments.count >= 2, pathPrefixes.contains(segments[0]) {
                candidate = segments[1]
            } else {
                candidate = nil
            }
        default:
            candidate = nil
        }
        guard let candidate, isValid(candidate) else { return nil }
        return candidate
    }

    /// `true` for an 11-character id made of letters, digits, `-` and `_`.
    public static func isValid(_ id: String) -> Bool {
        id.unicodeScalars.count == 11 && id.unicodeScalars.allSatisfy { scalar in
            switch scalar {
            case "a"..."z", "A"..."Z", "0"..."9", "-", "_": true
            default: false
            }
        }
    }

    private static let pathPrefixes: Set<String> = ["shorts", "embed", "live", "v"]
}
