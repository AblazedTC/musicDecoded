package com.musicdecoded.mobile.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PianoVoicingTest {
    private val voicings = Contracts.voicings()

    @Test fun matchesEveryCaseInTheContract() {
        val cases = Json.parseToJsonElement(Contracts.text("fixtures/piano-voicings.json")).jsonObject["cases"]!!.jsonArray
        assertTrue(cases.isNotEmpty())
        for (element in cases) {
            val case = element.jsonObject
            val label = case["label"]!!.jsonPrimitive.content
            val inversion = case["inversion"]!!.jsonPrimitive.int
            val transpose = case["transpose"]!!.jsonPrimitive.int
            val actual = PianoVoicing.of(voicings, label, inversion, transpose)
            val name = "$label inv $inversion transpose $transpose"
            val expected = case["expected"]
            if (expected == null || expected is JsonNull) {
                assertNull(name, actual)
                continue
            }
            val e = expected.jsonObject
            fun ints(key: String) = e[key]!!.jsonArray.map { it.jsonPrimitive.int }
            val expectedVoicing = PianoVoicing(
                inversions = e["inversions"]!!.jsonPrimitive.int,
                rightHand = ints("rightHand"),
                bass = e["bass"]!!.jsonPrimitive.int,
                rightHandFingers = ints("rightHandFingers"),
                bassFinger = e["bassFinger"]!!.jsonPrimitive.int,
            )
            assertEquals(name, expectedVoicing, actual)
        }
    }

    @Test fun everyChordFitsTheFixedKeyboard() {
        for (root in voicings.pitchClass.keys) for (quality in voicings.qualities.keys) for (inversion in 0..3) for (transpose in -6..6) {
            val v = checkNotNull(PianoVoicing.of(voicings, "$root:$quality", inversion, transpose))
            val all = v.rightHand + v.bass
            assertTrue("$root:$quality $inversion $transpose → $all", all.all { it in PianoVoicing.KEYBOARD_LOW..PianoVoicing.KEYBOARD_HIGH })
        }
    }

    @Test fun basicUsesTheTriadWithoutSlashBass() {
        val v = checkNotNull(PianoVoicing.of(voicings, "D:maj7/3", simplify = true))
        assertEquals(listOf(62, 66, 69), v.rightHand)
        assertEquals(50, v.bass)
    }
}
