import Foundation

/// Pulls a SoundCloud track link out of pasted or shared text (T-502).
///
/// Accepts `soundcloud.com/{user}/{track}` (also `www.` and `m.`, plus a private
/// `/{user}/{track}/s-{token}` link) and `on.soundcloud.com/{code}` short links, with or without a
/// scheme, anywhere in the text ("Listen to X on #SoundCloud https://on.soundcloud.com/abc").
/// The result is normalized: `https`, host `soundcloud.com` (short links keep `on.`), no query or
/// fragment, no trailing slash. Profiles, playlists (`/sets/`) and other pages return `nil`.
public enum SoundCloudURL {
    /// The first SoundCloud track URL in `text`, or `nil`.
    public static func parse(_ text: String) -> String? {
        for token in text.split(whereSeparator: { $0.isWhitespace }) {
            if let url = parseToken(String(token)) {
                return url
            }
        }
        return nil
    }

    /// A display title from the link's slug ("artist-name/my-song" → "My Song"), or `nil` for
    /// short links, whose slug is an opaque code.
    public static func suggestedTitle(for url: String) -> String? {
        guard let components = URLComponents(string: url), components.host == "soundcloud.com" else { return nil }
        let segments = components.path.split(separator: "/")
        guard segments.count >= 2 else { return nil }
        return humanize(String(segments[1]))
    }

    /// The uploader from the link's slug ("artist-name" → "Artist Name"), or `nil`.
    public static func suggestedArtist(for url: String) -> String? {
        guard let components = URLComponents(string: url), components.host == "soundcloud.com" else { return nil }
        guard let first = components.path.split(separator: "/").first else { return nil }
        return humanize(String(first))
    }

    private static func parseToken(_ raw: String) -> String? {
        // Shared text often wraps the link in punctuation.
        let token = raw.trimmingCharacters(in: CharacterSet(charactersIn: "()<>[]\"'.,;!"))
        guard token.isEmpty == false else { return nil }
        let urlText = token.contains("://") ? token : "https://" + token
        guard
            let components = URLComponents(string: urlText),
            let scheme = components.scheme?.lowercased(),
            scheme == "https" || scheme == "http",
            let rawHost = components.host?.lowercased()
        else { return nil }

        let segments = components.path.split(separator: "/").map(String.init)
        guard segments.allSatisfy(isSafeSegment) else { return nil }

        switch rawHost {
        case "on.soundcloud.com":
            guard segments.count == 1 else { return nil }
            return "https://on.soundcloud.com/" + segments[0]
        case "soundcloud.com", "www.soundcloud.com", "m.soundcloud.com":
            guard segments.count == 2 || (segments.count == 3 && segments[2].hasPrefix("s-")) else { return nil }
            guard reservedFirstSegments.contains(segments[0].lowercased()) == false,
                  reservedSecondSegments.contains(segments[1].lowercased()) == false
            else { return nil }
            return "https://soundcloud.com/" + segments.joined(separator: "/")
        default:
            return nil
        }
    }

    /// Letters, digits, `-` and `_` (SoundCloud slugs and short-link codes).
    private static func isSafeSegment(_ segment: String) -> Bool {
        segment.isEmpty == false && segment.unicodeScalars.allSatisfy { scalar in
            switch scalar {
            case "a"..."z", "A"..."Z", "0"..."9", "-", "_": true
            default: false
            }
        }
    }

    private static func humanize(_ slug: String) -> String? {
        let words = slug.split(whereSeparator: { $0 == "-" || $0 == "_" }).map { word in
            word.prefix(1).uppercased() + word.dropFirst()
        }
        let title = words.joined(separator: " ")
        return title.isEmpty ? nil : title
    }

    /// Site pages that look like `/{user}/{track}` but aren't tracks.
    private static let reservedFirstSegments: Set<String> = [
        "discover", "search", "you", "stream", "charts", "upload", "settings", "messages", "notifications", "pages",
    ]
    /// Profile tabs and playlists under a user.
    private static let reservedSecondSegments: Set<String> = [
        "sets", "albums", "tracks", "likes", "reposts", "popular-tracks", "followers", "following", "comments",
        "spotlight", "playlists",
    ]
}
