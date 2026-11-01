import Foundation

/// What the in-app YouTube browser (T-308i) reads from the page it's showing.
public enum YouTubeURL {
    /// The videoId when `url` is a single video page (`/watch?v=`, `/shorts/`, `youtu.be/`,
    /// `music.youtube.com/watch?v=`, extra params ignored); `nil` for the feed, search,
    /// playlists, channels and any other site.
    public static func videoId(from url: URL?) -> String? {
        guard let url else { return nil }
        return YouTubeVideoID.parse(url.absoluteString)
    }

    /// Whether the in-app browser may load a page on `host`: youtube.com and its subdomains,
    /// youtu.be, and Google's sign-in and consent pages. Other links open outside the app.
    public static func isAllowedInBrowser(host: String?) -> Bool {
        guard let host = host?.lowercased(), host.isEmpty == false else { return false }
        if host == "youtube.com" || host.hasSuffix(".youtube.com") { return true }
        return ["youtu.be", "accounts.google.com", "consent.google.com"].contains(host)
    }

    /// The video's title from the page's `document.title` ("(3) Song - YouTube" → "Song"), or
    /// `nil` when it's empty or just "YouTube".
    public static func videoTitle(fromPageTitle pageTitle: String?) -> String? {
        guard var title = pageTitle?.trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
        // The unread-notifications count YouTube prefixes, e.g. "(3) ".
        if title.hasPrefix("("), let close = title.firstIndex(of: ")"),
           title[title.index(after: title.startIndex)..<close].allSatisfy(\.isNumber) {
            title = String(title[title.index(after: close)...]).trimmingCharacters(in: .whitespaces)
        }
        for suffix in [" - YouTube Music", " - YouTube"] where title.hasSuffix(suffix) {
            title = String(title.dropLast(suffix.count)).trimmingCharacters(in: .whitespaces)
            break
        }
        guard title.isEmpty == false, title != "YouTube", title != "YouTube Music" else { return nil }
        return title
    }
}
