package com.musicdecoded.mobile.sources

import org.junit.Assert.assertEquals
import org.junit.Test

class InputLevelTest {
    @Test fun amplitudeMapsToDbfsClampedToTheMeterFloor() {
        assertEquals(0f, InputLevel.dbfs(32_767), 0.01f)
        assertEquals(-6.02f, InputLevel.dbfs(16_384), 0.01f)
        assertEquals(InputLevel.MIN_DB, InputLevel.dbfs(0))
        assertEquals(InputLevel.MIN_DB, InputLevel.dbfs(1)) // -90 dBFS reads as the floor
    }

    @Test fun sixteenSegmentsSpanTheMeterRange() {
        assertEquals(0, InputLevel.litSegments(InputLevel.MIN_DB))
        assertEquals(8, InputLevel.litSegments(-30f))
        assertEquals(16, InputLevel.litSegments(0f))
        assertEquals(0.5f, InputLevel.normalized(-30f), 0.001f)
    }

    @Test fun timerIsHoursMinutesSeconds() {
        assertEquals("00:00:00", InputLevel.formatTimer(999))
        assertEquals("00:01:23", InputLevel.formatTimer(83_000))
        assertEquals("00:10:00", InputLevel.formatTimer(600_000))
    }
}
