import Foundation
import Testing
@testable import MusicDecodedCore

struct YouTubeURLTests {
    @Test(arguments: [
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ&pp=ygUEc29uZw%3D%3D",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890&index=2",
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=42s",
        "https://m.youtube.com/shorts/dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ?feature=share",
        "https://youtu.be/dQw4w9WgXcQ?si=abcDEF123",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=RDAMVM",
    ])
    func findsTheVideoOnVideoPages(_ text: String) {
        #expect(YouTubeURL.videoId(from: URL(string: text)) == "dQw4w9WgXcQ")
    }

    @Test(arguments: [
        "https://m.youtube.com/",
        "https://m.youtube.com/results?search_query=sunny+days",
        "https://m.youtube.com/playlist?list=PL1234567890",
        "https://m.youtube.com/@SomeChannel",
        "https://m.youtube.com/@SomeChannel/videos",
        "https://m.youtube.com/channel/UC1234567890",
        "https://m.youtube.com/feed/subscriptions",
        "https://accounts.google.com/ServiceLogin",
        "about:blank",
    ])
    func noVideoElsewhere(_ text: String) {
        #expect(YouTubeURL.videoId(from: URL(string: text)) == nil)
    }

    @Test func noURL() {
        #expect(YouTubeURL.videoId(from: nil) == nil)
    }

    @Test(arguments: [
        ("Sunny Days - YouTube", "Sunny Days"),
        ("(3) Sunny Days - YouTube", "Sunny Days"),
        ("Sunny Days - YouTube Music", "Sunny Days"),
        ("  Sunny Days  ", "Sunny Days"),
        ("(Live) Sunny Days - YouTube", "(Live) Sunny Days"),
        ("Artist - Song - YouTube", "Artist - Song"),
    ])
    func cleansThePageTitle(_ pageTitle: String, _ expected: String) {
        #expect(YouTubeURL.videoTitle(fromPageTitle: pageTitle) == expected)
    }

    @Test(arguments: ["", "   ", "YouTube", "(12) YouTube", "YouTube Music"])
    func noTitleForGenericPages(_ pageTitle: String) {
        #expect(YouTubeURL.videoTitle(fromPageTitle: pageTitle) == nil)
    }

    @Test(arguments: [
        "youtube.com", "m.youtube.com", "www.youtube.com", "music.youtube.com", "M.YouTube.com",
        "youtu.be", "accounts.google.com", "consent.google.com",
    ])
    func allowsYouTubeAndSignIn(_ host: String) {
        #expect(YouTubeURL.isAllowedInBrowser(host: host))
    }

    @Test(arguments: [
        "", "google.com", "www.google.com", "evilyoutube.com", "youtube.com.evil.example",
        "example.com", "accounts.google.com.evil.example",
    ])
    func blocksOtherHosts(_ host: String) {
        #expect(YouTubeURL.isAllowedInBrowser(host: host) == false)
    }

    @Test func blocksNoHost() {
        #expect(YouTubeURL.isAllowedInBrowser(host: nil) == false)
    }

    @Test func noTitleWithoutAPage() {
        #expect(YouTubeURL.videoTitle(fromPageTitle: nil) == nil)
    }
}
