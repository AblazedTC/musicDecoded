package com.musicdecoded.mobile.library

import android.app.Application
import androidx.room.Room
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.auth.Session
import com.musicdecoded.mobile.auth.SessionStore
import com.musicdecoded.mobile.core.Contracts
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.network.ApiJson
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import com.musicdecoded.mobile.network.TranscriptionApiService
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.toByteArray
import io.ktor.client.request.HttpRequestData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
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
private const val PUT_URL = "http://storage.test/put/up-1"
private const val ANALYSIS_URL = "http://storage.test/analysis/track-1.json"
private val jsonHeaders = headersOf(HttpHeaders.ContentType, "application/json")

private class FixedSessionStore : SessionStore {
    override suspend fun session() = Session("token-1", "u1", Kind.GUEST, Plan.FREE)
    override suspend fun saveSession(session: Session) = Unit
    override suspend fun clearSession() = Unit
    override suspend fun deviceId() = "device-1"
    override suspend fun cachedConfig(): ClientConfig? = null
    override suspend fun saveConfig(config: ClientConfig) = Unit
}

private fun job(status: String, jobId: String = "job-1", error: String? = null, progress: Int = 0) =
    """{"jobId":"$jobId","trackId":"track-1","status":"$status","stage":null,"progress":$progress,"error":${error?.let { "\"$it\"" } ?: "null"}}"""

/** A fake API + object storage. Scripted per test; records every request. */
private class FakeServer {
    val requests = mutableListOf<HttpRequestData>()
    val bodies = mutableMapOf<String, String>()
    var uploadCount = 0
    var createJobStatus = HttpStatusCode.Accepted
    var createJobBody = job("queued")
    /** Responses for successive `GET /jobs/{id}`; the last one repeats. */
    val pollResponses = ArrayDeque<String>()

    val engine = MockEngine { request ->
        requests += request
        val path = "${request.method.value} ${request.url.host}${request.url.encodedPath}"
        when (path) {
            "POST api.test/uploads" -> {
                uploadCount++
                bodies["uploads"] = String(request.body.toByteArray())
                respond("""{"uploadId":"up-$uploadCount","putUrl":"$PUT_URL","expiresAt":"2026-01-01T00:00:00Z"}""", HttpStatusCode.Created, jsonHeaders)
            }
            "PUT storage.test/put/up-1" -> {
                bodies["put"] = String(request.body.toByteArray())
                respond("", HttpStatusCode.OK)
            }
            "POST api.test/jobs" -> {
                bodies["jobs"] = String(request.body.toByteArray())
                respond(createJobBody, createJobStatus, jsonHeaders)
            }
            "GET api.test/jobs/job-1", "GET api.test/jobs/job-2" -> {
                val body = if (pollResponses.size > 1) pollResponses.removeFirst() else pollResponses.first()
                respond(body, HttpStatusCode.OK, jsonHeaders)
            }
            "GET api.test/tracks/track-1/analysis" ->
                respond("""{"url":"$ANALYSIS_URL","schemaVersion":1}""", HttpStatusCode.OK, jsonHeaders)
            "GET storage.test/analysis/track-1.json" ->
                respond(Contracts.text("fixtures/sample.analysis.json"), HttpStatusCode.OK, jsonHeaders)
            else -> error("Unexpected request $path")
        }
    }

    fun count(method: String, path: String) =
        requests.count { it.method.value == method && it.url.encodedPath == path }
}

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class) // skip the app's launch-time network/session work
class TranscriptionRepositoryTest {
    @get:Rule val folder = TemporaryFolder()

    private val db = Room.inMemoryDatabaseBuilder(RuntimeEnvironment.getApplication(), LibraryDatabase::class.java)
        .allowMainThreadQueries()
        .build()
    private val dao = db.libraryDao()
    private val server = FakeServer()
    private val scheduled = mutableListOf<Pair<String, Boolean>>()
    private val now = Instant.parse("2026-10-29T12:00:00Z")

    private lateinit var filesDir: File
    private lateinit var repository: TranscriptionRepository

    private fun setUp() {
        filesDir = folder.newFolder("files")
        val http = createHttpClient(server.engine, debugLogging = false)
        val auth = AuthRepository(AuthApiService(http, BASE), ConfigApiService(http, BASE), FixedSessionStore())
        repository = TranscriptionRepository(
            TranscriptionApiService(http, BASE),
            auth,
            dao,
            AnalysisStore(File(filesDir, "analysis")),
            filesDir,
            { trackId, replace -> scheduled += trackId to replace },
            clock = { now },
        )
    }

    @After fun close() = db.close()

    private fun pendingAudio(bytes: String = "abc"): LocalAudio {
        val file = repository.newPendingFile("m4a").apply { writeText(bytes) }
        return LocalAudio(file, "My song", Source.FILE, durationSec = 12.0)
    }

    private fun body(name: String): JsonObject = ApiJson.parseToJsonElement(server.bodies.getValue(name)).jsonObject

    @Test fun completeImmediatelyUploadsThenSavesTheAnalysis() = runTest {
        setUp()
        server.createJobStatus = HttpStatusCode.OK
        server.createJobBody = job("complete", progress = 100)
        server.pollResponses += job("complete", progress = 100)

        val trackId = repository.start(pendingAudio())

        assertEquals("track-1", trackId)
        // POST /uploads: content type, size and lowercase hex SHA-256 of the bytes.
        assertEquals("audio/mp4", body("uploads")["contentType"]!!.jsonPrimitive.content)
        assertEquals("3", body("uploads")["sizeBytes"]!!.jsonPrimitive.content)
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            body("uploads")["sha256"]!!.jsonPrimitive.content,
        )
        // PUT: raw bytes, same Content-Type, no bearer.
        val put = server.requests.single { it.method.value == "PUT" }
        assertEquals("abc", server.bodies["put"])
        assertEquals("audio/mp4", put.body.contentType.toString())
        assertNull(put.headers[HttpHeaders.Authorization])
        // POST /jobs references the upload.
        assertEquals("file", body("jobs")["source"]!!.jsonPrimitive.content)
        assertEquals("up-1", body("jobs")["uploadId"]!!.jsonPrimitive.content)
        assertEquals("My song", body("jobs")["title"]!!.jsonPrimitive.content)
        // Item is analyzing, the audio is renamed to audio/{trackId}.{ext}, polling is scheduled.
        val analyzing = dao.get("track-1")!!
        assertEquals(LibraryStatus.ANALYZING, analyzing.status)
        assertEquals("job-1", analyzing.jobId)
        assertEquals("up-1", analyzing.audioUploadId)
        assertEquals("audio/track-1.m4a", analyzing.localAudioPath)
        assertTrue(File(filesDir, "audio/track-1.m4a").isFile)
        assertEquals(listOf("track-1" to true), scheduled)

        repository.poll(trackId)

        val ready = dao.get("track-1")!!
        assertEquals(LibraryStatus.READY, ready.status)
        assertNull(ready.jobId)
        assertEquals("D major", ready.key)
        assertEquals(136.1, ready.bpm!!, 1e-9)
        assertEquals(263.407, ready.durationSec!!, 1e-9)
        assertTrue(File(filesDir, "analysis/track-1.json").isFile)
        assertTrue(repository.jobs.value.isEmpty())
    }

    @Test fun queuedThenProcessingThenCompletePollsEveryTwoSeconds() = runTest {
        setUp()
        server.pollResponses += listOf(job("queued"), job("processing", progress = 40), job("complete", progress = 100))

        val trackId = repository.start(pendingAudio())
        assertEquals(LibraryStatus.ANALYZING, dao.get(trackId)!!.status)
        repository.poll(trackId)

        assertEquals(3, server.count("GET", "/jobs/job-1"))
        assertEquals(4_000L, testScheduler.currentTime)
        assertEquals(LibraryStatus.READY, dao.get(trackId)!!.status)
    }

    @Test fun pollingBacksOffToTenSecondsAfterSixtySeconds() = runTest {
        setUp()
        repeat(35) { server.pollResponses += job("processing") }
        server.pollResponses += job("complete")

        repository.poll(repository.start(pendingAudio()))

        // 30 polls 2 s apart reach 60 s, then 5 more 10 s apart.
        assertEquals(60_000L + 5 * 10_000L, testScheduler.currentTime)
        assertEquals(36, server.count("GET", "/jobs/job-1"))
    }

    @Test fun failedJobMarksTheItemFailedAndRetryReusesTheUpload() = runTest {
        setUp()
        server.pollResponses += job("failed", error = "Audio too short")

        val trackId = repository.start(pendingAudio())
        repository.poll(trackId)

        val failed = dao.get(trackId)!!
        assertEquals(LibraryStatus.FAILED, failed.status)
        assertEquals("Audio too short", failed.error)
        assertNull(failed.jobId)

        server.createJobBody = job("queued", jobId = "job-2")
        server.pollResponses.clear()
        server.pollResponses += job("complete", jobId = "job-2")
        repository.retry(trackId)

        assertEquals(1, server.uploadCount) // no second POST /uploads
        assertEquals(1, server.count("PUT", "/put/up-1"))
        assertEquals("up-1", body("jobs")["uploadId"]!!.jsonPrimitive.content)
        val retried = dao.get(trackId)!!
        assertEquals(LibraryStatus.ANALYZING, retried.status)
        assertEquals("job-2", retried.jobId)
        assertNull(retried.error)
        assertEquals("track-1" to true, scheduled.last())

        repository.poll(trackId)
        assertEquals(LibraryStatus.READY, dao.get(trackId)!!.status)
    }

    @Test fun rateLimitedShowsAClearMessageAndCreatesNoItem() = runTest {
        setUp()
        server.createJobStatus = HttpStatusCode.TooManyRequests
        server.createJobBody = """{"code":"rate_limited","message":"Daily limit reached"}"""

        try {
            repository.start(pendingAudio())
            fail("Expected a TranscriptionException")
        } catch (e: TranscriptionException) {
            assertEquals(TranscriptionRepository.RATE_LIMITED, e.message)
        }
        assertNull(dao.get("track-1"))
        assertTrue(scheduled.isEmpty())
    }

    @Test fun unsupportedFormatFailsBeforeAnyRequest() = runTest {
        setUp()
        val file = repository.newPendingFile("txt").apply { writeText("abc") }
        try {
            repository.start(LocalAudio(file, "Notes", Source.FILE))
            fail("Expected a TranscriptionException")
        } catch (e: TranscriptionException) {
            assertTrue(e.message!!.contains("isn't supported"))
        }
        assertTrue(server.requests.isEmpty())
    }

    @Test fun resumePendingReschedulesRunningJobsWithoutReplacing() = runTest {
        setUp()
        repository.start(pendingAudio())
        scheduled.clear()

        repository.resumePending()

        assertEquals(listOf("track-1" to false), scheduled)
        assertFalse(dao.analyzing().isEmpty())
    }
    @Test fun youTubeJobSkipsTheUploadAndPollsTheSameWay() = runTest {
        setUp()
        server.pollResponses += listOf(job("processing", progress = 50), job("complete", progress = 100))

        val trackId = repository.startYouTube("dQw4w9WgXcQ", "Sunny Days", "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg", 200.0)

        assertEquals("track-1", trackId)
        assertEquals(0, server.count("POST", "/uploads"))
        assertTrue(server.requests.none { it.method.value == "PUT" })
        assertEquals("youtube", body("jobs")["source"]!!.jsonPrimitive.content)
        assertEquals("dQw4w9WgXcQ", body("jobs")["videoId"]!!.jsonPrimitive.content)
        assertNull(body("jobs")["uploadId"])
        val analyzing = dao.get(trackId)!!
        assertEquals(LibraryStatus.ANALYZING, analyzing.status)
        assertEquals(Source.YOUTUBE, analyzing.source)
        assertEquals("dQw4w9WgXcQ", analyzing.sourceRef)
        assertEquals("Sunny Days", analyzing.title)
        assertEquals("https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg", analyzing.thumbnailUrl)
        // Streaming source: no local audio is kept (PLAN.md R-1).
        assertNull(analyzing.localAudioPath)
        assertNull(analyzing.audioUploadId)
        assertEquals(listOf("track-1" to true), scheduled)

        repository.poll(trackId)

        assertEquals(LibraryStatus.READY, dao.get(trackId)!!.status)
        assertEquals(2, server.count("GET", "/jobs/job-1"))
    }

    @Test fun youTubeRetryRequestsByVideoIdAgain() = runTest {
        setUp()
        server.pollResponses += job("failed", error = "Video unavailable")
        val trackId = repository.startYouTube("dQw4w9WgXcQ", "Sunny Days", null, null)
        repository.poll(trackId)
        assertEquals(LibraryStatus.FAILED, dao.get(trackId)!!.status)

        server.createJobBody = job("queued", jobId = "job-2")
        repository.retry(trackId)

        assertEquals(2, server.count("POST", "/jobs"))
        assertEquals(0, server.count("POST", "/uploads"))
        assertEquals("dQw4w9WgXcQ", body("jobs")["videoId"]!!.jsonPrimitive.content)
        assertEquals("job-2", dao.get(trackId)!!.jobId)
        assertEquals(LibraryStatus.ANALYZING, dao.get(trackId)!!.status)
    }

    @Test fun youTubePlanLimitShowsTheUpsellMessage() = runTest {
        setUp()
        server.createJobStatus = HttpStatusCode.TooManyRequests
        server.createJobBody = """{"code":"rate_limited","message":"Daily limit"}"""
        try {
            repository.startYouTube("dQw4w9WgXcQ", "Sunny Days", null, null)
            fail("Expected TranscriptionException")
        } catch (e: TranscriptionException) {
            assertEquals(TranscriptionRepository.RATE_LIMITED, e.message)
        }
        assertNull(dao.get("track-1"))
    }

    @Test fun soundCloudJobSendsTheUrlAndKeepsNoAudio() = runTest {
        setUp()
        server.pollResponses += job("complete", progress = 100)
        val url = "https://soundcloud.com/artist/sunny-days"

        val trackId = repository.startSoundCloud(url, "Sunny days")

        assertEquals(0, server.count("POST", "/uploads"))
        assertEquals("soundcloud", body("jobs")["source"]!!.jsonPrimitive.content)
        assertEquals(url, body("jobs")["soundcloudUrl"]!!.jsonPrimitive.content)
        assertNull(body("jobs")["videoId"])
        val analyzing = dao.get(trackId)!!
        assertEquals(Source.SOUNDCLOUD, analyzing.source)
        assertEquals(url, analyzing.sourceRef)
        assertEquals("Sunny days", analyzing.title)
        assertNull(analyzing.localAudioPath) // PLAN.md R-1
        assertEquals(listOf("track-1" to true), scheduled)

        repository.poll(trackId)
        assertEquals(LibraryStatus.READY, dao.get(trackId)!!.status)
    }

    @Test fun soundCloudRetryRequestsByUrlAgain() = runTest {
        setUp()
        server.pollResponses += job("failed", error = "Track unavailable")
        val url = "https://soundcloud.com/artist/sunny-days"
        val trackId = repository.startSoundCloud(url, "Sunny days")
        repository.poll(trackId)

        server.createJobBody = job("queued", jobId = "job-2")
        repository.retry(trackId)

        assertEquals(2, server.count("POST", "/jobs"))
        assertEquals(url, body("jobs")["soundcloudUrl"]!!.jsonPrimitive.content)
        assertEquals("job-2", dao.get(trackId)!!.jobId)
    }

    @Test fun soundCloudFeatureDisabledShowsAFriendlyMessage() = runTest {
        setUp()
        server.createJobStatus = HttpStatusCode.Forbidden
        server.createJobBody = """{"code":"feature_disabled","message":"soundcloud is disabled"}"""
        try {
            repository.startSoundCloud("https://soundcloud.com/artist/sunny-days", "Sunny days")
            fail("Expected TranscriptionException")
        } catch (e: TranscriptionException) {
            assertEquals(TranscriptionRepository.SOUNDCLOUD_DISABLED, e.message)
        }
        assertNull(dao.get("track-1"))
    }
}
