package com.musicdecoded.mobile.core

import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class AnalysisTest {
    @Test fun decodesSampleFixture() {
        val analysis = Contracts.sampleAnalysis()

        assertEquals(1, analysis.schemaVersion)
        assertEquals("fixture-bad", analysis.meta.trackId)
        assertEquals(Source.FILE, analysis.meta.source)
        assertNull(analysis.meta.sourceRef)
        assertEquals(263.407, analysis.meta.durationSec, 0.0)
        assertEquals(Analysis.Key("D", Analysis.Mode.MAJOR, 0.0), analysis.key)
        assertEquals(136.1, analysis.bpm!!, 0.0)
        assertEquals(Analysis.TimeSignature(4, 4), analysis.timeSignature)
        assertEquals(581, analysis.beats.size)
        assertEquals(94, analysis.chords.size)
        assertEquals("N", analysis.chords.first().label)
        assertEquals(Analysis.Chord(0.72, 1.138, "A:maj"), analysis.chords[1])
        assertNull(analysis.spectrogram)
        assertNull(analysis.waveformPeaks)
    }

    @Test fun fixtureChordsAreSortedAndContiguous() {
        Contracts.sampleAnalysis().chords.zipWithNext { previous, next ->
            assertEquals("gap or overlap at ${previous.end}", previous.end, next.start, 0.0)
        }
    }

    @Test fun decodesNullableFieldsAndVisuals() {
        val json = """
        {
          "schemaVersion": 1,
          "meta": { "trackId": "t1", "source": "youtube", "sourceRef": "abc123",
                    "durationSec": 3, "analyzedAt": "2026-01-01T00:00:00Z", "analyzerVersion": "1.0.0" },
          "key": null, "bpm": null, "timeSignature": null,
          "beats": [{ "t": 0.5, "downbeat": true }],
          "chords": [{ "start": 0, "end": 3, "label": "C:maj", "confidence": 0.9 }],
          "spectrogram": { "fps": 20, "bins": 2, "frames": 1, "encoding": "uint8-base64", "data": "AP8=" },
          "waveformPeaks": { "fps": 20, "peaks": [0, 0.5, 1] }
        }
        """
        val analysis = Analysis.decode(json)

        assertEquals(Source.YOUTUBE, analysis.meta.source)
        assertEquals("abc123", analysis.meta.sourceRef)
        assertNull(analysis.key)
        assertNull(analysis.bpm)
        assertNull(analysis.timeSignature)
        assertEquals(0.9, analysis.chords.first().confidence!!, 0.0)
        assertEquals(2, analysis.spectrogram?.bins)
        assertEquals(listOf(0.0, 0.5, 1.0), analysis.waveformPeaks?.peaks)
    }

    @Test fun roundTripsThroughJson() {
        val original = Contracts.sampleAnalysis()
        assertEquals(original, Analysis.decode(original.encode()))
    }

    @Test fun encodesRequiredNullableKeysAsNull() {
        val analysis = SyntheticAnalysis.make().copy(bpm = null, timeSignature = null)
        val obj = AnalysisJson.parseToJsonElement(analysis.encode()).jsonObject

        assertEquals(JsonNull, obj["key"])
        assertEquals(JsonNull, obj["bpm"])
        assertEquals(JsonNull, obj["timeSignature"])
        assertFalse("optional visuals are omitted, not null", obj.containsKey("spectrogram"))
    }
}
