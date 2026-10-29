package com.musicdecoded.mobile.auth

import com.musicdecoded.mobile.network.ApiException
import com.musicdecoded.mobile.network.ApiJson
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.AuthResponse
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.toByteArray
import io.ktor.client.request.HttpRequestData
import io.ktor.client.request.HttpResponseData
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

private class InMemorySessionStore(var session: Session? = null) : SessionStore {
    var config: ClientConfig? = null
    var generatedIds = 0
    private var id: String? = null

    override suspend fun session() = session
    override suspend fun saveSession(session: Session) { this.session = session }
    override suspend fun clearSession() { session = null }
    override suspend fun deviceId() = id ?: "device-${++generatedIds}".also { id = it }
    override suspend fun cachedConfig() = config
    override suspend fun saveConfig(config: ClientConfig) { this.config = config }
}

private const val BASE = "http://api.test"
private val json = headersOf(HttpHeaders.ContentType, "application/json")

private fun authJson(token: String, kind: String = "guest") =
    """{"token":"$token","userId":"u1","kind":"$kind","plan":"free"}"""

private const val CONFIG_JSON =
    """{"offlineStreamAudio":false,"maxUploadBytes":104857600,"soundcloudEnabled":true,"plan":"free","limits":{"dailyJobs":10,"jobsUsedToday":2}}"""

private const val UNAUTHORIZED = """{"code":"unauthorized","message":"Token expired"}"""

class AuthRepositoryTest {
    private val requests = mutableListOf<HttpRequestData>()

    private fun repository(
        store: SessionStore,
        handler: suspend MockRequestHandleScope.(HttpRequestData) -> HttpResponseData,
    ): AuthRepository {
        val engine = MockEngine { request ->
            requests += request
            handler(request)
        }
        val http = createHttpClient(engine, debugLogging = false)
        configApi = ConfigApiService(http, BASE)
        return AuthRepository(AuthApiService(http, BASE), configApi, store)
    }

    private lateinit var configApi: ConfigApiService

    private fun paths() = requests.map { "${it.method.value} ${it.url.encodedPath}" }

    @Test fun noTokenSignsInAsGuestWithTheDeviceId() = runTest {
        val store = InMemorySessionStore()
        val repo = repository(store) { respond(authJson("guest-1"), HttpStatusCode.OK, json) }

        val session = repo.ensureSession()

        assertEquals(Session("guest-1", "u1", Kind.GUEST, Plan.FREE), session)
        assertEquals(session, store.session)
        assertEquals(listOf("POST /auth/anonymous"), paths())
        val body = ApiJson.parseToJsonElement(String(requests.single().body.toByteArray())).jsonObject
        assertEquals("device-1", body["deviceId"]?.jsonPrimitive?.content)
        assertEquals("android", body["platform"]?.jsonPrimitive?.content)

        repo.ensureSession()
        assertEquals("stored token is reused", 1, requests.size)
    }

    @Test fun deviceIdIsGeneratedOnceAndReusedForReAuth() = runTest {
        val store = InMemorySessionStore()
        val repo = repository(store) { respond(authJson("guest-1"), HttpStatusCode.OK, json) }
        repo.ensureSession()
        store.session = null
        repo.ensureSession()
        assertEquals(1, store.generatedIds)
    }

    @Test fun startFetchesAndCachesConfigWithTheBearerToken() = runTest {
        val store = InMemorySessionStore(Session("t0", "u1", Kind.GUEST, Plan.FREE))
        val repo = repository(store) { respond(CONFIG_JSON, HttpStatusCode.OK, json) }

        repo.start()

        assertEquals(listOf("GET /config"), paths())
        assertEquals("Bearer t0", requests.single().headers[HttpHeaders.Authorization])
        val expected = ClientConfig(false, 104_857_600, true, Plan.FREE, ClientConfig.Limits(10, 2))
        assertEquals(expected, repo.config.value)
        assertEquals(expected, store.config)
    }

    @Test fun configFailureFallsBackToTheCachedValue() = runTest {
        val cached = ClientConfig(false, 1, false, Plan.FREE, ClientConfig.Limits(null, 0))
        val store = InMemorySessionStore(Session("t0", "u1", Kind.GUEST, Plan.FREE)).apply { config = cached }
        val repo = repository(store) { respond("""{"code":"internal","message":"boom"}""", HttpStatusCode.InternalServerError, json) }

        assertEquals(cached, repo.refreshConfig())
        assertEquals(cached, repo.config.value)
    }

    @Test fun guest401ReAuthsOnceThenRetries() = runTest {
        val store = InMemorySessionStore(Session("expired", "u1", Kind.GUEST, Plan.FREE))
        val repo = repository(store) { request ->
            when {
                request.url.encodedPath == "/auth/anonymous" -> respond(authJson("fresh"), HttpStatusCode.OK, json)
                request.headers[HttpHeaders.Authorization] == "Bearer fresh" -> respond(CONFIG_JSON, HttpStatusCode.OK, json)
                else -> respond(UNAUTHORIZED, HttpStatusCode.Unauthorized, json)
            }
        }

        repo.refreshConfig()

        assertEquals(listOf("GET /config", "POST /auth/anonymous", "GET /config"), paths())
        assertEquals("fresh", store.session?.token)
        assertEquals(10, repo.config.value?.limits?.dailyJobs)
    }

    @Test fun onlyOneRetryWhenTheFreshTokenIsAlsoRejected() = runTest {
        val store = InMemorySessionStore(Session("expired", "u1", Kind.GUEST, Plan.FREE))
        val repo = repository(store) { request ->
            if (request.url.encodedPath == "/auth/anonymous") respond(authJson("fresh"), HttpStatusCode.OK, json)
            else respond(UNAUTHORIZED, HttpStatusCode.Unauthorized, json)
        }

        try {
            repo.authorized { configApi.config(it) }
            fail("expected ApiException")
        } catch (e: ApiException) {
            assertEquals(401, e.status)
            assertEquals("unauthorized", e.error.code)
        }
        assertEquals("exactly one re-auth", listOf("GET /config", "POST /auth/anonymous", "GET /config"), paths())
    }

    @Test fun account401IsNotSilentlyDowngradedToGuest() = runTest {
        val store = InMemorySessionStore(Session("acct", "u1", Kind.ACCOUNT, Plan.FREE))
        val repo = repository(store) { respond(UNAUTHORIZED, HttpStatusCode.Unauthorized, json) }

        assertNull(repo.refreshConfig())
        assertEquals(listOf("GET /config"), paths())
        assertEquals("acct", store.session?.token)
    }

    @Test fun loginSendsTheGuestTokenToLinkData() = runTest {
        val store = InMemorySessionStore(Session("guest-1", "u1", Kind.GUEST, Plan.FREE))
        val repo = repository(store) { respond(authJson("acct", kind = "account"), HttpStatusCode.OK, json) }

        repo.login("a@b.co", "password1")

        assertEquals(listOf("POST /auth/login"), paths())
        assertEquals("Bearer guest-1", requests.single().headers[HttpHeaders.Authorization])
        assertEquals(Kind.ACCOUNT, store.session?.kind)
    }

    @Test fun loginErrorCarriesTheContractError() = runTest {
        val repo = repository(InMemorySessionStore()) {
            respond("""{"code":"unauthorized","message":"Invalid email or password"}""", HttpStatusCode.Unauthorized, json)
        }
        try {
            repo.login("a@b.co", "wrongpass")
            fail("expected ApiException")
        } catch (e: ApiException) {
            assertEquals("Invalid email or password", e.message)
        }
        assertEquals("a 401 on login never triggers guest re-auth", listOf("POST /auth/login"), paths())
    }

    @Test fun logoutFallsBackToAGuestSession() = runTest {
        val store = InMemorySessionStore(Session("acct", "u1", Kind.ACCOUNT, Plan.PRO))
        val repo = repository(store) { respond(authJson("guest-2"), HttpStatusCode.OK, json) }

        val session = repo.logout()

        assertEquals(Kind.GUEST, session.kind)
        assertTrue(paths() == listOf("POST /auth/anonymous"))
    }

    @Test fun decodesContractSamples() {
        assertEquals(
            AuthResponse("t", "u1", Kind.ACCOUNT, Plan.PRO),
            ApiJson.decodeFromString(AuthResponse.serializer(), """{"token":"t","userId":"u1","kind":"account","plan":"pro","extra":1}"""),
        )
        val config = ApiJson.decodeFromString(
            ClientConfig.serializer(),
            """{"offlineStreamAudio":true,"maxUploadBytes":5,"plan":"pro","limits":{"dailyJobs":null,"jobsUsedToday":0}}""",
        )
        assertEquals(false, config.soundcloudEnabled)
        assertNull(config.limits.dailyJobs)
    }
}
