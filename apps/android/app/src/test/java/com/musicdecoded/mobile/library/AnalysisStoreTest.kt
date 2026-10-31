package com.musicdecoded.mobile.library

import com.musicdecoded.mobile.core.Contracts
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class AnalysisStoreTest {
    @get:Rule val temp = TemporaryFolder()

    private val fixtureJson = Contracts.text("fixtures/sample.analysis.json")
    private fun store() = AnalysisStore(File(temp.root, "analysis"))

    @Test fun savesVerbatimAndDecodesTheFixture() = runTest {
        val store = store()
        store.save("fixture-bad", fixtureJson)

        assertEquals(fixtureJson, File(temp.root, "analysis/fixture-bad.json").readText())
        assertEquals(Contracts.sampleAnalysis(), store.load("fixture-bad"))
        assertTrue(store.exists("fixture-bad"))
    }

    @Test fun overwritesAnExistingFile() = runTest {
        val store = store()
        store.save("t1", fixtureJson)
        store.save("t1", fixtureJson)
        assertEquals(94, store.load("t1")?.chords?.size)
    }

    @Test fun missingFileLoadsAsNull() = runTest {
        assertNull(store().load("nope"))
        assertFalse(store().exists("nope"))
    }

    @Test fun refusesInvalidJsonAndLeavesNothingBehind() = runTest {
        val store = store()
        try {
            store.save("t1", """{"schemaVersion":1}""")
            fail("expected a decode error")
        } catch (_: IllegalArgumentException) {
        }
        assertFalse(store.exists("t1"))
    }

    @Test fun rejectsPathTraversal() = runTest {
        try {
            store().load("../secrets")
            fail("expected IllegalArgumentException")
        } catch (_: IllegalArgumentException) {
        }
    }

    @Test fun deleteRemovesTheFile() = runTest {
        val store = store()
        store.save("t1", fixtureJson)
        store.delete("t1")
        assertFalse(store.exists("t1"))
    }
}
