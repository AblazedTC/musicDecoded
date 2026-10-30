package com.musicdecoded.mobile.sources.soundcloud

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SoundCloudUrlTest {
    @Test fun plainTrackLink() {
        assertEquals("https://soundcloud.com/artist/track-name", SoundCloudUrl.extract("https://soundcloud.com/artist/track-name"))
    }

    @Test fun shareSheetSentenceWithTrackingParameters() {
        val shared = "Listen to Sunny Days by Artist on #SoundCloud\nhttps://soundcloud.com/artist/sunny-days?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing"
        assertEquals("https://soundcloud.com/artist/sunny-days", SoundCloudUrl.extract(shared))
    }

    @Test fun shortShareLink() {
        assertEquals("https://on.soundcloud.com/AbC123xyz", SoundCloudUrl.extract("Check this out: https://on.soundcloud.com/AbC123xyz"))
    }

    @Test fun mobileAndWwwHostsAreNormalized() {
        assertEquals("https://soundcloud.com/a/b", SoundCloudUrl.extract("https://m.soundcloud.com/a/b"))
        assertEquals("https://soundcloud.com/a/b", SoundCloudUrl.extract("http://www.soundcloud.com/a/b#t=0:30"))
    }

    @Test fun withoutSchemeAndWithTrailingPunctuation() {
        assertEquals("https://soundcloud.com/artist/track", SoundCloudUrl.extract("try soundcloud.com/artist/track."))
        assertEquals("https://soundcloud.com/artist/track", SoundCloudUrl.extract("(https://soundcloud.com/artist/track)"))
    }

    @Test fun privateTrackKeepsItsSecretToken() {
        assertEquals("https://soundcloud.com/artist/track/s-AbCdE", SoundCloudUrl.extract("https://soundcloud.com/artist/track/s-AbCdE"))
    }

    @Test fun firstSoundCloudLinkWins() {
        val text = "https://example.com/x then https://soundcloud.com/a/first and https://soundcloud.com/b/second"
        assertEquals("https://soundcloud.com/a/first", SoundCloudUrl.extract(text))
    }

    @Test fun skipsAProfileLinkForALaterTrack() {
        assertEquals("https://soundcloud.com/a/track", SoundCloudUrl.extract("soundcloud.com/a and soundcloud.com/a/track"))
    }

    @Test fun rejectsNonTracksAndOtherHosts() {
        assertNull(SoundCloudUrl.extract(""))
        assertNull(SoundCloudUrl.extract("no link here"))
        assertNull(SoundCloudUrl.extract("https://soundcloud.com/"))
        assertNull(SoundCloudUrl.extract("https://soundcloud.com/artist")) // a profile
        assertNull(SoundCloudUrl.extract("https://on.soundcloud.com/"))
        assertNull(SoundCloudUrl.extract("https://notsoundcloud.com/a/b"))
        assertNull(SoundCloudUrl.extract("https://api.soundcloud.com/tracks/123"))
        assertNull(SoundCloudUrl.extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    }

    @Test fun titleFromSlug() {
        assertEquals("Sunny days", SoundCloudUrl.titleFrom("https://soundcloud.com/artist/sunny-days"))
        assertEquals("SoundCloud track", SoundCloudUrl.titleFrom("https://on.soundcloud.com/AbC123"))
    }
}
