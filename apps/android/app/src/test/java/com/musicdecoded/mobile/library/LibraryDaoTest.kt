package com.musicdecoded.mobile.library

import android.app.Application
import androidx.room.Room
import com.musicdecoded.mobile.core.Contracts
import com.musicdecoded.mobile.core.Source
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.time.Instant

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class) // skip the app's launch-time network/session work
class LibraryDaoTest {
    private val db = Room.inMemoryDatabaseBuilder(RuntimeEnvironment.getApplication(), LibraryDatabase::class.java)
        .allowMainThreadQueries()
        .build()
    private val dao = db.libraryDao()

    @After fun close() = db.close()

    private fun item(id: String, createdAt: Long) = LibraryItem(
        trackId = id,
        title = "Song $id",
        source = Source.FILE,
        createdAt = Instant.ofEpochSecond(createdAt),
        updatedAt = Instant.ofEpochSecond(createdAt),
        status = LibraryStatus.ANALYZING,
        jobId = "job-$id",
        localAudioPath = "audio/$id.m4a",
    )

    @Test fun upsertAndGetRoundTripEveryField() = runTest {
        val original = item("a", 100).copy(
            artist = "Artist", sourceRef = null, thumbnailUrl = "https://x/y.jpg", audioUploadId = "up1",
            favorite = true, key = "B minor", bpm = 140.0, lastPlayedAt = Instant.ofEpochSecond(150),
            syncedAt = Instant.ofEpochSecond(160),
        )
        dao.upsert(original)
        assertEquals(original, dao.get("a"))

        val updated = original.copy(title = "Renamed", status = LibraryStatus.FAILED)
        dao.upsert(updated)
        assertEquals(updated, dao.get("a"))
        assertNull(dao.get("missing"))
    }

    @Test fun newestFirstSortsByCreatedAtAndHidesTombstones() = runTest {
        dao.upsert(item("old", 100))
        dao.upsert(item("new", 300))
        dao.upsert(item("mid", 200))
        assertEquals(listOf("new", "mid", "old"), dao.newestFirst().first().map { it.trackId })

        dao.markDeleted("mid", Instant.ofEpochSecond(400))
        assertEquals(listOf("new", "old"), dao.newestFirst().first().map { it.trackId })
    }

    @Test fun deleteIsATombstone() = runTest {
        dao.upsert(item("a", 100))
        dao.markDeleted("a", Instant.ofEpochSecond(500))

        val row = dao.get("a")!!
        assertTrue(row.deleted)
        assertEquals(Instant.ofEpochSecond(500), row.updatedAt)
    }

    @Test fun favoriteBumpsUpdatedAt() = runTest {
        dao.upsert(item("a", 100))
        dao.setFavorite("a", true, Instant.ofEpochSecond(200))

        val row = dao.get("a")!!
        assertTrue(row.favorite)
        assertEquals(Instant.ofEpochSecond(200), row.updatedAt)
    }

    @Test fun withAnalysisCopiesListMetadataFromTheFixture() = runTest {
        val ready = item("fixture-bad", 100).withAnalysis(Contracts.sampleAnalysis(), Instant.ofEpochSecond(200))
        dao.upsert(ready)

        val row = dao.get("fixture-bad")!!
        assertEquals("D major", row.key)
        assertEquals(136.1, row.bpm!!, 0.0)
        assertEquals(263.407, row.durationSec!!, 0.0)
        assertEquals(LibraryStatus.READY, row.status)
        assertNull(row.jobId)
    }

    @Test fun recentIsLimitedNewestFirst() = runTest {
        (1..5).forEach { dao.upsert(item("t$it", it * 100L)) }
        dao.markDeleted("t5", Instant.ofEpochSecond(600))
        assertEquals(listOf("t4", "t3", "t2"), dao.recent(3).first().map { it.trackId })
    }

    private suspend fun search(text: String = "", favorites: Boolean = false, offline: Boolean = false, source: Source? = null) =
        dao.search(likePattern(text), favorites, offline, source?.name).first().map { it.trackId }

    @Test fun searchMatchesTitleArtistAndKeyAndFilters() = runTest {
        dao.upsert(item("a", 100).copy(title = "Somehow", artist = "DAY6", key = "F# minor", favorite = true))
        dao.upsert(item("b", 200).copy(title = "Blue 100%", artist = "Someone", key = "D major", localAudioPath = null))
        dao.upsert(item("c", 300).copy(title = "Voice memo", source = Source.RECORDING, key = "Bb major"))
        dao.upsert(item("gone", 400).copy(title = "Somehow too"))
        dao.markDeleted("gone", Instant.ofEpochSecond(500))

        assertEquals(listOf("c", "b", "a"), search())
        assertEquals(listOf("a"), search("someh"))
        assertEquals(listOf("b", "a"), search("some")) // title or artist
        assertEquals(listOf("a"), search("f# min"))
        assertEquals(listOf("c"), search("bb"))
        assertEquals(listOf("b"), search("100%")) // % is literal
        assertEquals(emptyList<String>(), search("_"))
        assertEquals(listOf("a"), search(favorites = true))
        assertEquals(listOf("c", "a"), search(offline = true))
        assertEquals(listOf("c"), search(source = Source.RECORDING))
    }

    @Test fun countsDecodedAndOffline() = runTest {
        assertEquals(LibraryCounts(0, 0), dao.counts().first())
        dao.upsert(item("a", 100).copy(status = LibraryStatus.READY))
        dao.upsert(item("b", 200).copy(status = LibraryStatus.READY, localAudioPath = null))
        dao.upsert(item("c", 300)) // analyzing, local
        dao.upsert(item("d", 400).copy(status = LibraryStatus.READY))
        dao.markDeleted("d", Instant.ofEpochSecond(500))
        assertEquals(LibraryCounts(decoded = 2, offline = 2), dao.counts().first())
    }
}
