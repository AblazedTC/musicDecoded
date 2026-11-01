package com.musicdecoded.mobile.sources.youtube

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class YouTubeUrlTest {
    private val id = "dQw4w9WgXcQ"

    @Test fun parsesEverySupportedLinkShape() {
        listOf(
            "https://www.youtube.com/watch?v=$id",
            "https://youtube.com/watch?v=$id&t=42s",
            "https://www.youtube.com/watch?feature=share&v=$id",
            "http://m.youtube.com/watch?v=$id",
            "https://music.youtube.com/watch?v=$id&list=RDAMVM$id",
            "https://youtu.be/$id",
            "https://youtu.be/$id?si=abc123&t=10",
            "https://www.youtube.com/shorts/$id",
            "https://youtube.com/shorts/$id?feature=share",
            "https://www.youtube.com/embed/$id",
            "https://www.youtube-nocookie.com/embed/$id?start=3",
            "https://www.youtube.com/live/$id",
            "youtube.com/watch?v=$id",
            "youtu.be/$id",
            "  https://youtu.be/$id  ",
            id,
            // What the in-app browser (m.youtube.com) reports while navigating.
            "https://m.youtube.com/watch?v=$id&pp=ygUFaGVsbG8%3D",
            "https://m.youtube.com/watch?v=$id&list=PL123&index=2",
            "https://m.youtube.com/shorts/$id",
            "https://music.youtube.com/watch?v=$id&si=xyz",
        ).forEach { assertEquals(it, id, YouTubeUrl.videoId(it)) }
    }

    @Test fun rejectsAnythingElse() {
        listOf(
            "",
            "hello world",
            "https://example.com/watch?v=$id",
            "https://vimeo.com/123456",
            "https://www.youtube.com/watch?v=short",
            "https://www.youtube.com/watch",
            "https://www.youtube.com/@channel",
            "https://www.youtube.com/playlist?list=PL123",
            "https://youtu.be/",
            "ftp://youtu.be/$id",
            "javascript:alert(1)",
            "https://notyoutube.com/watch?v=$id",
            "https://youtube.com.evil.com/watch?v=$id",
            "https://m.youtube.com/",
            "https://m.youtube.com/results?search_query=sunny+days",
            "https://m.youtube.com/playlist?list=PL123",
            "https://m.youtube.com/@channel/videos",
            "https://m.youtube.com/channel/UC1234567890",
            "https://m.youtube.com/feed/subscriptions",
        ).forEach { assertNull(it, YouTubeUrl.videoId(it)) }
    }

    @Test fun pageTitleDropsTheYouTubeSuffix() {
        assertEquals("Sunny Days", YouTubeUrl.pageTitle("Sunny Days - YouTube"))
        assertEquals("A - B", YouTubeUrl.pageTitle(" A - B "))
        assertNull(YouTubeUrl.pageTitle("YouTube"))
        assertNull(YouTubeUrl.pageTitle(" "))
        assertNull(YouTubeUrl.pageTitle(null))
    }
}
