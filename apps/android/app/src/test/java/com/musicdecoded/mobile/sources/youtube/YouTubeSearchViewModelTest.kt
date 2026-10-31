package com.musicdecoded.mobile.sources.youtube

import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.auth.Session
import com.musicdecoded.mobile.auth.SessionStore
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import com.musicdecoded.mobile.network.YouTubeApiService
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockEngineConfig
import io.ktor.client.engine.mock.respond
import io.ktor.client.request.HttpRequestData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.IOException

private const val BASE = "http://api.test"
private val jsonHeaders = headersOf(HttpHeaders.ContentType, "application/json")

private class FixedSessionStore : SessionStore {
    override suspend fun session() = Session("token-1", "u1", Kind.GUEST, Plan.FREE)
    override suspend fun saveSession(session: Session) = Unit
    override suspend fun clearSession() = Unit
    override suspend fun deviceId() = "device-1"
    override suspend fun cachedConfig(): ClientConfig? = null
    override suspend fun saveConfig(config: ClientConfig) = Unit
}

private class FakeRecent(var saved: List<String> = emptyList()) : RecentQueries {
    override suspend fun load() = saved
    override suspend fun save(queries: List<String>) {
        saved = queries
    }
}

private fun video(id: String) =
    """{"videoId":"$id","title":"Title $id","channelTitle":"Channel","thumbnailUrl":"https://i.ytimg.com/vi/$id/mqdefault.jpg","durationSec":200}"""

private fun page(vararg ids: String, next: String? = null) =
    """{"items":[${ids.joinToString(",") { video(it) }}],"nextPageToken":${next?.let { "\"$it\"" } ?: "null"}}"""

/** `GET /youtube/search` against a MockEngine (T-304a). */
@OptIn(ExperimentalCoroutinesApi::class)
class YouTubeSearchViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    private val requests = mutableListOf<HttpRequestData>()

    /** The response for a request (by q / pageToken); default: one page of two videos. */
    private var respond: (q: String?, pageToken: String?) -> Pair<HttpStatusCode, String> =
        { _, _ -> HttpStatusCode.OK to page("aaaaaaaaaaa", "bbbbbbbbbbb", next = "p2") }
    private var offline = false
    private val recent = FakeRecent()

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    private fun viewModel(): YouTubeSearchViewModel {
        // The engine runs on the test dispatcher, so virtual time covers the whole request.
        val engine = MockEngine(
            MockEngineConfig().apply {
                dispatcher = this@YouTubeSearchViewModelTest.dispatcher
                addHandler { request ->
                    requests += request
                    if (offline) throw IOException("offline")
                    val (status, body) = respond(request.url.parameters["q"], request.url.parameters["pageToken"])
                    respond(body, status, jsonHeaders)
                }
            },
        )
        val http = createHttpClient(engine, debugLogging = false)
        val auth = AuthRepository(AuthApiService(http, BASE), ConfigApiService(http, BASE), FixedSessionStore())
        return YouTubeSearchViewModel(auth, YouTubeApiService(http, BASE), recent)
    }

    private fun searches() = requests.filter { it.url.encodedPath == "/youtube/search" }

    private fun TestScope.type(vm: YouTubeSearchViewModel, text: String) {
        vm.onQueryChange(text)
        runCurrent()
    }

    @Test fun typingIsDebouncedFourHundredMilliseconds() = runTest(dispatcher) {
        val vm = viewModel()
        type(vm, "s")
        advanceTimeBy(200)
        type(vm, "sunny")
        advanceTimeBy(200)
        type(vm, "sunny days")
        advanceTimeBy(399)
        runCurrent()
        assertTrue(searches().isEmpty())

        advanceTimeBy(2)
        advanceUntilIdle()

        val request = searches().single()
        assertEquals("sunny days", request.url.parameters["q"])
        assertNull(request.url.parameters["pageToken"])
        assertEquals("Bearer token-1", request.headers[HttpHeaders.Authorization])
        val state = vm.state.value
        assertEquals("sunny days", state.resultsFor)
        assertEquals(listOf("aaaaaaaaaaa", "bbbbbbbbbbb"), state.results.map { it.videoId })
        assertEquals(200, state.results.first().durationSec)
        assertEquals("p2", state.nextPageToken)
        assertFalse(state.loading)
    }

    @Test fun successfulQueriesBecomeRecentChipsNewestFirstMaxEight() = runTest(dispatcher) {
        recent.saved = listOf("old")
        val vm = viewModel()
        advanceUntilIdle()
        assertEquals(listOf("old"), vm.state.value.recent)

        (1..9).forEach { vm.submit("q$it"); advanceUntilIdle() }
        vm.submit("Q3"); advanceUntilIdle()

        val expected = listOf("Q3", "q9", "q8", "q7", "q6", "q5", "q4", "q2")
        assertEquals(expected, vm.state.value.recent)
        assertEquals(expected, recent.saved)
    }

    @Test fun theSameQueryIsNotFetchedTwice() = runTest(dispatcher) {
        val vm = viewModel()
        vm.submit("sunny days"); advanceUntilIdle()
        vm.onQueryChange("sunny days  "); advanceUntilIdle()
        vm.submit("sunny days"); advanceUntilIdle()
        assertEquals(1, searches().size)
    }

    @Test fun clearingTheQueryClearsResults() = runTest(dispatcher) {
        val vm = viewModel()
        vm.submit("sunny"); advanceUntilIdle()
        vm.onQueryChange(""); advanceUntilIdle()
        assertNull(vm.state.value.resultsFor)
        assertTrue(vm.state.value.results.isEmpty())
        assertEquals(1, searches().size)
    }

    @Test fun infiniteScrollAppendsTheNextPageUntilThereIsNone() = runTest(dispatcher) {
        respond = { _, token ->
            HttpStatusCode.OK to when (token) {
                null -> page("aaaaaaaaaaa", "bbbbbbbbbbb", next = "p2")
                "p2" -> page("bbbbbbbbbbb", "ccccccccccc", next = "p3")
                else -> page("ddddddddddd")
            }
        }
        val vm = viewModel()
        vm.submit("sunny"); advanceUntilIdle()
        vm.loadMore(); advanceUntilIdle()
        vm.loadMore(); advanceUntilIdle()

        val state = vm.state.value
        // Duplicates across pages are dropped.
        assertEquals(listOf("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"), state.results.map { it.videoId })
        assertNull(state.nextPageToken)
        assertFalse(state.canLoadMore)
        assertEquals(listOf(null, "p2", "p3"), searches().map { it.url.parameters["pageToken"] })
        assertTrue(searches().all { it.url.parameters["q"] == "sunny" })

        vm.loadMore(); advanceUntilIdle()
        assertEquals(3, searches().size)
    }

    @Test fun featureDisabledMeansSearchUnavailable() = runTest(dispatcher) {
        respond = { _, _ -> HttpStatusCode.Forbidden to """{"code":"feature_disabled","message":"YouTube search is not configured"}""" }
        val vm = viewModel()
        vm.submit("sunny"); advanceUntilIdle()
        assertEquals(SearchError.UNAVAILABLE, vm.state.value.error)
        assertFalse(SearchError.UNAVAILABLE.retryable)
        assertTrue(vm.state.value.recent.isEmpty())
    }

    @Test fun quotaIsNotThePlanUpsell() = runTest(dispatcher) {
        respond = { _, _ -> HttpStatusCode.TooManyRequests to """{"code":"rate_limited","message":"quota"}""" }
        val vm = viewModel()
        vm.submit("sunny"); advanceUntilIdle()
        val error = vm.state.value.error
        assertEquals(SearchError.QUOTA, error)
        val copy = "${error!!.title} ${error.message}".lowercase()
        assertFalse(copy.contains("pro"))
        assertFalse(copy.contains("upgrade"))
        assertFalse(copy.contains("plan"))
    }

    @Test fun offlineThenRetrySucceeds() = runTest(dispatcher) {
        offline = true
        val vm = viewModel()
        vm.submit("sunny"); advanceUntilIdle()
        assertEquals(SearchError.OFFLINE, vm.state.value.error)

        offline = false
        vm.retry(); advanceUntilIdle()
        assertNull(vm.state.value.error)
        assertEquals(2, vm.state.value.results.size)
    }

    @Test fun aFailedNextPageKeepsTheListAndCanBeRetried() = runTest(dispatcher) {
        val vm = viewModel()
        vm.submit("sunny"); advanceUntilIdle()
        offline = true
        vm.loadMore(); advanceUntilIdle()
        assertEquals(SearchError.OFFLINE, vm.state.value.loadMoreError)
        assertEquals(2, vm.state.value.results.size)
        assertFalse(vm.state.value.canLoadMore)

        offline = false
        respond = { _, _ -> HttpStatusCode.OK to page("ccccccccccc") }
        vm.retryLoadMore(); advanceUntilIdle()
        assertNull(vm.state.value.loadMoreError)
        assertEquals(3, vm.state.value.results.size)
    }

    @Test fun pastedUrlOpensTheVideoWithoutSearching() = runTest(dispatcher) {
        val vm = viewModel()
        vm.openPastedUrl("https://example.com/nope")
        assertTrue(vm.state.value.pasteError != null)
        assertNull(vm.state.value.openVideo)

        vm.openPastedUrl("https://youtu.be/dQw4w9WgXcQ?si=x")
        val video = vm.state.value.openVideo!!
        assertEquals("dQw4w9WgXcQ", video.videoId)
        assertEquals("https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg", video.thumbnailUrl)
        assertNull(vm.state.value.pasteError)
        vm.videoOpened()
        assertNull(vm.state.value.openVideo)
        advanceUntilIdle()
        assertTrue(searches().isEmpty())
    }

    @Test fun recentQueriesHelper() {
        assertEquals(listOf("b", "a"), RecentQueries.add(listOf("a", "B"), "b"))
        assertEquals("a b", YouTubeSearchViewModel.normalize("  a \n b "))
        assertEquals(200, YouTubeSearchViewModel.normalize("x".repeat(300)).length)
    }
}
