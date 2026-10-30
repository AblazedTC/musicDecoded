import Testing
@testable import MusicDecodedCore

struct SoundCloudURLTests {
    @Test(arguments: [
        "https://soundcloud.com/artist-name/my-song",
        "http://soundcloud.com/artist-name/my-song",
        "https://www.soundcloud.com/artist-name/my-song",
        "https://m.soundcloud.com/artist-name/my-song",
        "soundcloud.com/artist-name/my-song",
        "https://soundcloud.com/artist-name/my-song/",
        "https://soundcloud.com/artist-name/my-song?si=abc123&utm_source=clipboard",
        "https://soundcloud.com/artist-name/my-song#t=1:00",
        "  https://soundcloud.com/artist-name/my-song\n",
        "HTTPS://SOUNDCLOUD.COM/artist-name/my-song",
        "Listen to My Song by Artist Name on #SoundCloud https://soundcloud.com/artist-name/my-song",
        "(https://soundcloud.com/artist-name/my-song).",
    ])
    func parsesTrackLinks(_ text: String) {
        #expect(SoundCloudURL.parse(text) == "https://soundcloud.com/artist-name/my-song")
    }

    @Test func keepsShortLinks() {
        #expect(SoundCloudURL.parse("https://on.soundcloud.com/AbC123xyz") == "https://on.soundcloud.com/AbC123xyz")
        #expect(SoundCloudURL.parse("Check this https://on.soundcloud.com/AbC123xyz?si=1") == "https://on.soundcloud.com/AbC123xyz")
    }

    @Test func keepsPrivateTokens() {
        #expect(SoundCloudURL.parse("https://soundcloud.com/artist/song/s-AbCdE") == "https://soundcloud.com/artist/song/s-AbCdE")
    }

    @Test(arguments: [
        "",
        "   ",
        "my-song",
        "https://soundcloud.com/",
        "https://soundcloud.com/artist-name",
        "https://soundcloud.com/artist-name/sets/my-playlist",
        "https://soundcloud.com/artist-name/likes",
        "https://soundcloud.com/discover/sets",
        "https://soundcloud.com/search/sounds",
        "https://soundcloud.com/a/b/c",
        "https://on.soundcloud.com/",
        "https://on.soundcloud.com/a/b",
        "https://notsoundcloud.com/artist/song",
        "https://soundcloud.com.evil.example/artist/song",
        "https://api.soundcloud.com/tracks/123",
        "ftp://soundcloud.com/artist/song",
        "https://soundcloud.com/artist/so%20ng",
        "https://youtu.be/dQw4w9WgXcQ",
    ])
    func rejectsEverythingElse(_ text: String) {
        #expect(SoundCloudURL.parse(text) == nil)
    }

    @Test func suggestsTitleAndArtistFromTheSlug() {
        let url = "https://soundcloud.com/artist-name/my-great_song"
        #expect(SoundCloudURL.suggestedTitle(for: url) == "My Great Song")
        #expect(SoundCloudURL.suggestedArtist(for: url) == "Artist Name")
        #expect(SoundCloudURL.suggestedTitle(for: "https://on.soundcloud.com/AbC123") == nil)
        #expect(SoundCloudURL.suggestedArtist(for: "https://on.soundcloud.com/AbC123") == nil)
    }
}
