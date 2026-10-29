package com.musicdecoded.mobile.ui.components

import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Contracts
import org.junit.Assert.assertEquals
import org.junit.Test

class FormattingTest {
    @Test fun fixtureMetaLine() {
        val analysis = Contracts.sampleAnalysis()
        assertEquals("4:23 · D major · 136 BPM", songMetaLine(analysis.meta.durationSec, analysis.key, analysis.bpm))
    }

    @Test fun skipsUnknownParts() {
        assertEquals("0:05", songMetaLine(5.9, null, null))
        assertEquals("F♯ minor · 90 BPM", songMetaLine(null, Analysis.Key("F#", Analysis.Mode.MINOR), 89.6))
        assertEquals("", songMetaLine(null, null, null))
    }
}
