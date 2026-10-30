package com.musicdecoded.mobile.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RomanNumeralTest {
    @Serializable private data class Fixture(val cases: List<Case>)

    @Serializable private data class Case(val key: Analysis.Key? = null, val label: String, val numeral: String? = null)

    @Test fun matchesEveryCaseInTheSharedFixture() {
        val fixture = Json { ignoreUnknownKeys = true }
            .decodeFromString(Fixture.serializer(), Contracts.text("fixtures/roman-numerals.json"))
        assertTrue(fixture.cases.size >= 20)
        for (case in fixture.cases) {
            assertEquals("${case.key} ${case.label}", case.numeral, romanNumeral(case.label, case.key))
        }
    }

    @Test fun minorKeyIntervalsAndQualities() {
        val aMinor = Analysis.Key("A", Analysis.Mode.MINOR)
        assertEquals("♯vii°", romanNumeral("G#:dim7", aMinor))
        assertEquals("♯III", romanNumeral("C#:maj", aMinor))
        assertEquals("i", romanNumeral("A:minmaj7", aMinor))
        assertNull(romanNumeral("X", aMinor))
    }
}
