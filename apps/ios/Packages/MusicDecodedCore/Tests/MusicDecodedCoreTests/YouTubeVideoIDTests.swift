import Testing
@testable import MusicDecodedCore

struct YouTubeVideoIDTests {
    @Test(arguments: [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        "http://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "youtube.com/watch?v=dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abcDEF123",
        "youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ?feature=share",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=RDAMVM",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "  https://youtu.be/dQw4w9WgXcQ\n",
        "HTTPS://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
    ])
    func parsesSupportedLinks(_ text: String) {
        #expect(YouTubeVideoID.parse(text) == "dQw4w9WgXcQ")
    }

    @Test func keepsDashesAndUnderscores() {
        #expect(YouTubeVideoID.parse("https://youtu.be/a-b_c-d_e-f") == "a-b_c-d_e-f")
    }

    @Test(arguments: [
        "",
        "   ",
        "dQw4w9WgXcQ",
        "never gonna give you up",
        "https://vimeo.com/123456789",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQextra",
        "https://www.youtube.com/channel/UC1234567890",
        "https://youtu.be/",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgX%20Q",
    ])
    func rejectsEverythingElse(_ text: String) {
        #expect(YouTubeVideoID.parse(text) == nil)
    }
}
