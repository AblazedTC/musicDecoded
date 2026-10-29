package com.musicdecoded.mobile.library

import android.app.Application
import androidx.room.Room
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.auth.Session
import com.musicdecoded.mobile.auth.SessionStore
import com.musicdecoded.mobile.core.Contracts
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.SyncRepository.Companion.toSyncItem
import com.musicdecoded.mobile.network.ApiJson
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import com.musicdecoded.mobile.network.SyncApiService
import com.musicdecoded.mobile.network.SyncItem
import com.musicdecoded.mobile.network.SyncItems
import com.musicdecoded.mobile.network.SyncPage
import com.musicdecoded.mobile.network.TranscriptionApiService
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.toByteArray
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.File
import java.time.Instant

private const val BASE = "http://api.test"
private val jsonHeaders = headersOf(HttpHeaders.ContentType, "application/json")
private const val AUDIO_BYTES = "fake-mp3-bytes"

/**
 * An in-memory `/sync/library` with the server rules from T-402: LWW on updatedAt (ties keep the
 * server), pulls ordered by a server sequence, 500 per page, the cursor = last sequence seen.
 */
private class FakeSyncServer {
    private data class Row(val item: SyncItem, val seq: Long)

    private val rows = linkedMapOf<String, Row>()
    private var seq = 0L
    var analysisAvailable = true
    val calls = mutableListOf<String>()

    fun items(): List<SyncItem> = rows.values.map { it.item }
    fun item(trackId: String): SyncItem? = rows[trackId]?.item

    /** Writes as if another device had pushed [item]. */
    fun put(item: SyncItem) {
        val existing = rows[item.trackId]
        if (existing == null || Instant.parse(item.updatedAt) > Instant.parse(existing.item.updatedAt)) {
            rows[item.trackId] = Row(item, ++seq)
        }
    }

    val engine = MockEngine { request ->
        val path = request.url.encodedPath
        calls += "${request.method.value} $path"
        when {
            path == "/sync/library" && request.method.value == "GET" -> {
                val since = request.url.parameters["since"]?.toLong() ?: 0L
                val page = rows.values.filter { it.seq > since }.sortedBy { it.seq }.take(500)
                val cursor = page.lastOrNull()?.seq?.toString() ?: since.toString()
                respond(ApiJson.encodeToString(SyncPage.serializer(), SyncPage(page.map { it.item }, cursor)), HttpStatusCode.OK, jsonHeaders)
            }
            path == "/sync/library" && request.method.value == "PUT" -> {
                val pushed = ApiJson.decodeFromString(SyncItems.serializer(), String(request.body.toByteArray())).items
                check(pushed.size <= 500) { "Push over the contract limit" }
                pushed.forEach(::put)
                val state = SyncItems(pushed.map { rows.getValue(it.trackId).item })
                respond(ApiJson.encodeToString(SyncItems.serializer(), state), HttpStatusCode.OK, jsonHeaders)
            }
            path.startsWith("/tracks/") && path.endsWith("/analysis") -> {
                val trackId = path.removePrefix("/tracks/").removeSuffix("/analysis")
                if (analysisAvailable) {
                    respond("""{"url":"http://storage.test/analysis/$trackId.json","schemaVersion":1}""", HttpStatusCode.OK, jsonHeaders)
                } else {
                    respond("""{"code":"not_found","message":"No analysis"}""", HttpStatusCode.NotFound, jsonHeaders)
                }
            }
            path.startsWith("/analysis/") -> respond(Contracts.text("fixtures/sample.analysis.json"), HttpStatusCode.OK, jsonHeaders)
            path.startsWith("/uploads/") -> {
                val id = path.removePrefix("/uploads/")
                respond("""{"url":"http://storage.test/blob/$id","contentType":"audio/mpeg","expiresAt":"2026-01-01T00:00:00Z"}""", HttpStatusCode.OK, jsonHeaders)
            }
            path.startsWith("/blob/") -> respond(AUDIO_BYTES, HttpStatusCode.OK)
            else -> error("Unexpected request ${request.method.value} $path")
        }
    }
}

private class FixedSession(private val session: Session?) : SessionStore {
    override suspend fun session() = session
    override suspend fun saveSession(session: Session) = Unit
    override suspend fun clearSession() = Unit
    override suspend fun deviceId() = "device"
    override suspend fun cachedConfig(): ClientConfig? = null
    override suspend fun saveConfig(config: ClientConfig) = Unit
}

private class MemorySyncState : SyncStateStore {
    var user: String? = null
    var savedCursor: String? = null
    override suspend fun userId() = user
    override suspend fun cursor() = savedCursor
    override suspend fun save(userId: String, cursor: String?) {
        user = userId
        savedCursor = cursor
    }
}

/** One phone: its own Room database, files and sync state, talking to the shared fake server. */
private class Device(server: FakeSyncServer, root: File, kind: Kind = Kind.ACCOUNT) {
    val db = Room.inMemoryDatabaseBuilder(RuntimeEnvironment.getApplication(), LibraryDatabase::class.java)
        .allowMainThreadQueries()
        .build()
    val dao = db.libraryDao()
    val filesDir = root
    val analysisStore = AnalysisStore(File(root, "analysis"))
    val state = MemorySyncState()
    val sync: SyncRepository

    init {
        val http = createHttpClient(server.engine, debugLogging = false)
        val auth = AuthRepository(
            AuthApiService(http, BASE),
            ConfigApiService(http, BASE),
            FixedSession(Session("token", "user-1", kind, Plan.FREE)),
        )
        sync = SyncRepository(SyncApiService(http, BASE), TranscriptionApiService(http, BASE), auth, dao, analysisStore, filesDir, state)
    }
}

private fun at(seconds: Long): Instant = Instant.parse("2026-10-29T12:00:00Z").plusSeconds(seconds)

private fun item(
    trackId: String,
    updatedAt: Instant = at(0),
    source: Source = Source.YOUTUBE,
    title: String = "Song $trackId",
    favorite: Boolean = false,
    deleted: Boolean = false,
    audioUploadId: String? = null,
    localAudioPath: String? = null,
) = LibraryItem(
    trackId = trackId,
    title = title,
    source = source,
    favorite = favorite,
    createdAt = at(0),
    updatedAt = updatedAt,
    deleted = deleted,
    audioUploadId = audioUploadId,
    localAudioPath = localAudioPath,
    status = LibraryStatus.READY,
)

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
class SyncRepositoryTest {
    @get:Rule val folder = TemporaryFolder()

    private val server = FakeSyncServer()
    private val devices = mutableListOf<Device>()

    private fun device(kind: Kind = Kind.ACCOUNT) = Device(server, folder.newFolder(), kind).also { devices += it }

    @After fun close() = devices.forEach { it.db.close() }

    /** The synced fields of every row, for comparing devices (duration is refreshed locally from the analysis). */
    private suspend fun Device.snapshot(): Map<String, SyncItem> =
        server.items().mapNotNull { dao.get(it.trackId) }.associate { it.trackId to it.toSyncItem().copy(durationSec = null) }

    @Test fun twoDevicesConverge() = runTest {
        val a = device()
        val b = device()
        a.dao.upsert(item("t1", at(1), title = "From A"))
        b.dao.upsert(item("t2", at(2), title = "From B"))

        a.sync.sync()
        b.sync.sync()
        a.sync.sync()

        assertEquals(setOf("t1", "t2"), server.items().map { it.trackId }.toSet())
        assertEquals(a.snapshot(), b.snapshot())
        assertEquals("From B", a.dao.get("t2")?.title)
        // Pulled rows got their analysis file and are playable.
        assertTrue(a.analysisStore.exists("t2"))
        assertEquals(LibraryStatus.READY, a.dao.get("t2")?.status)
        assertTrue(b.analysisStore.exists("t1"))
    }

    @Test fun tombstonesPropagate() = runTest {
        val a = device()
        val b = device()
        a.dao.upsert(item("t1", at(1)))
        a.sync.sync()
        b.sync.sync()
        assertFalse(b.dao.get("t1")!!.deleted)

        a.dao.markDeleted("t1", at(5))
        a.sync.sync()
        b.sync.sync()

        assertTrue(server.item("t1")!!.deleted)
        assertTrue(b.dao.get("t1")!!.deleted)
    }

    @Test fun newerRemoteWinsOlderRemoteLoses() = runTest {
        val a = device()
        a.dao.upsert(item("newer", at(10), favorite = false))
        a.dao.upsert(item("older", at(10), favorite = false))
        a.sync.sync()

        // Another device: a newer edit of one row, an older edit of the other.
        server.put(item("newer", at(20), favorite = true).toSyncItem())
        server.put(item("older", at(5), favorite = true).toSyncItem()) // rejected by the server too
        // And a local edit, newer than the server, made offline.
        a.dao.setFavorite("older", true, at(30))
        a.sync.sync()

        assertTrue(a.dao.get("newer")!!.favorite)
        assertEquals(at(20), a.dao.get("newer")!!.updatedAt)
        assertEquals(at(30), Instant.parse(server.item("older")!!.updatedAt))
        assertTrue(server.item("older")!!.favorite)
    }

    @Test fun localRowNewerThanAPulledOneIsKept() = runTest {
        val a = device()
        server.put(item("t1", at(5), title = "Server title").toSyncItem())
        a.dao.upsert(item("t1", at(9), title = "Local title").copy(syncedAt = at(9)))

        a.sync.sync()

        assertEquals("Local title", a.dao.get("t1")!!.title)
    }

    @Test fun pullPaginatesUntilAShortPageAndStoresTheCursor() = runTest {
        (1..1201).forEach { server.put(item("t$it", at(it.toLong()), deleted = true).toSyncItem()) }
        val a = device()

        a.sync.sync()

        assertEquals(3, server.calls.count { it == "GET /sync/library" })
        assertEquals(1201, (1..1201).count { a.dao.get("t$it") != null })
        assertEquals("1201", a.state.savedCursor)

        // Next sync starts from the cursor: one empty page.
        server.calls.clear()
        a.sync.sync()
        assertEquals(listOf("GET /sync/library"), server.calls)
    }

    @Test fun pushBatchesOf500() = runTest {
        server.analysisAvailable = false
        val a = device()
        (1..501).forEach { a.dao.upsert(item("t$it", at(it.toLong()))) }

        a.sync.sync()

        assertEquals(2, server.calls.count { it == "PUT /sync/library" })
        assertEquals(501, server.items().size)
        assertTrue(a.dao.pendingPush(10).isEmpty())
    }

    @Test fun ownAudioIsDownloadedStreamingAudioIsNot() = runTest {
        server.put(item("rec", at(1), source = Source.RECORDING, audioUploadId = "up-1").toSyncItem())
        server.put(item("yt", at(1), source = Source.YOUTUBE).toSyncItem())
        val a = device()

        a.sync.sync()

        val path = a.dao.get("rec")!!.localAudioPath
        assertEquals("audio/rec.mp3", path)
        assertEquals(AUDIO_BYTES, File(a.filesDir, path!!).readText())
        assertNull(a.dao.get("yt")!!.localAudioPath)
        assertEquals(1, server.calls.count { it.startsWith("GET /uploads/") })
    }

    @Test fun missingAnalysisMarksTheRowFailedAndRetriesNextSync() = runTest {
        server.analysisAvailable = false
        server.put(item("t1", at(1)).toSyncItem())
        val a = device()

        a.sync.sync()
        assertEquals(LibraryStatus.FAILED, a.dao.get("t1")!!.status)

        server.analysisAvailable = true
        a.sync.sync()
        assertEquals(LibraryStatus.READY, a.dao.get("t1")!!.status)
        assertTrue(a.analysisStore.exists("t1"))
    }

    @Test fun guestDoesNothing() = runTest {
        val guest = device(Kind.GUEST)
        guest.dao.upsert(item("t1", at(1)))

        assertFalse(guest.sync.sync())
        assertTrue(server.calls.isEmpty())
    }
}
