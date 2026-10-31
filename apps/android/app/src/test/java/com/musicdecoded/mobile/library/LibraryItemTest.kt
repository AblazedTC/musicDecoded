package com.musicdecoded.mobile.library

import com.musicdecoded.mobile.core.Source
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant

class LibraryItemTest {
    private val then = Instant.parse("2026-10-01T00:00:00Z")
    private val now = Instant.parse("2026-10-30T00:00:00Z")

    private fun item(source: Source, title: String, sourceRef: String? = null, thumbnailUrl: String? = null, durationSec: Double? = null) =
        LibraryItem(
            trackId = "t", title = title, source = source, sourceRef = sourceRef, thumbnailUrl = thumbnailUrl,
            durationSec = durationSec, createdAt = then, updatedAt = then, status = LibraryStatus.ANALYZING,
        )

    @Test fun placeholderTitles() {
        assertTrue(item(Source.YOUTUBE, "YouTube video").hasPlaceholderTitle)
        assertTrue(item(Source.SOUNDCLOUD, "Sunny days", "https://soundcloud.com/artist/sunny-days").hasPlaceholderTitle)
        assertTrue(item(Source.SOUNDCLOUD, "SoundCloud track", "https://on.soundcloud.com/AbC").hasPlaceholderTitle)
        assertFalse(item(Source.YOUTUBE, "Sunny Days").hasPlaceholderTitle)
        assertFalse(item(Source.SOUNDCLOUD, "Sunny Days (Live)", "https://soundcloud.com/artist/sunny-days").hasPlaceholderTitle)
        assertFalse(item(Source.FILE, "YouTube video").hasPlaceholderTitle)
    }

    @Test fun jobMetadataReplacesPlaceholdersAndBumpsUpdatedAt() {
        val updated = item(Source.YOUTUBE, "YouTube video").withJobMetadata("Real title", "https://t/1.jpg", 213.0, now)
        assertEquals("Real title", updated.title)
        assertEquals("https://t/1.jpg", updated.thumbnailUrl)
        assertEquals(213.0, updated.durationSec!!, 0.0)
        assertEquals(now, updated.updatedAt)
    }

    @Test fun jobMetadataFillsOnlyWhatIsMissing() {
        val real = item(Source.YOUTUBE, "Sunny Days", thumbnailUrl = "https://mine.jpg", durationSec = 200.0)
        assertSame(real, real.withJobMetadata("Other", "https://other.jpg", 100.0, now))
        assertSame(real, real.withJobMetadata(null, null, null, now))

        val noThumb = item(Source.YOUTUBE, "Sunny Days").withJobMetadata("Other", "https://t/1.jpg", null, now)
        assertEquals("Sunny Days", noThumb.title)
        assertEquals("https://t/1.jpg", noThumb.thumbnailUrl)
        assertEquals(now, noThumb.updatedAt)
    }
}
