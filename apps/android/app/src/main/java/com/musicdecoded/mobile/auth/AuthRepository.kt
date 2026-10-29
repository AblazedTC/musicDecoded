package com.musicdecoded.mobile.auth

import com.musicdecoded.mobile.network.ApiException
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.AuthResponse
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.Kind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Guest session + client config (PLAN.md §3.4, D-2).
 * - No token → `POST /auth/anonymous` with this install's deviceId.
 * - [authorized] runs an API call with the bearer token; a 401 on a guest token re-authenticates
 *   once and retries once.
 * - `/config` is fetched at launch and cached; [config] falls back to the cached value offline.
 */
class AuthRepository(
    private val authApi: AuthApiService,
    private val configApi: ConfigApiService,
    private val store: SessionStore,
) {
    private val mutex = Mutex()
    private val _config = MutableStateFlow<ClientConfig?>(null)
    val config: StateFlow<ClientConfig?> = _config.asStateFlow()

    /** Called once at launch: make sure there's a session, then refresh the config. */
    suspend fun start() {
        _config.value = _config.value ?: store.cachedConfig()
        ensureSession()
        refreshConfig()
    }

    /** The stored session, or a new guest session. */
    suspend fun ensureSession(): Session = mutex.withLock { store.session() ?: signInAsGuest() }

    /** Runs [call] with the current token; on a 401 for a guest, re-auths once and retries once. */
    suspend fun <T> authorized(call: suspend (token: String) -> T): T {
        val session = ensureSession()
        try {
            return call(session.token)
        } catch (e: ApiException) {
            // Account tokens have no refresh endpoint in the contract yet; let the caller handle it.
            if (e.status != 401 || session.kind != Kind.GUEST) throw e
        }
        val fresh = mutex.withLock {
            // Another caller may already have re-authenticated while we waited.
            store.session()?.takeIf { it.token != session.token } ?: signInAsGuest()
        }
        return call(fresh.token)
    }

    /** Fetches `/config` and caches it. On failure, keeps (and returns) the last cached value. */
    suspend fun refreshConfig(): ClientConfig? {
        try {
            val fresh = authorized { configApi.config(it) }
            store.saveConfig(fresh)
            _config.value = fresh
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            _config.value = _config.value ?: store.cachedConfig()
        }
        return _config.value
    }

    /** Logs in; a guest token is sent so the server merges the guest's data into the account. */
    suspend fun login(email: String, password: String) {
        val response = authApi.login(email, password, guestToken())
        store.saveSession(response.toSession())
    }

    /** Creates an account, linking the guest's data. */
    suspend fun signup(email: String, password: String) {
        val response = authApi.signup(email, password, guestToken())
        store.saveSession(response.toSession())
    }

    /** Signs out of an account and falls back to this device's guest identity. */
    suspend fun logout(): Session {
        mutex.withLock { store.clearSession() }
        return ensureSession()
    }

    private suspend fun guestToken(): String? = store.session()?.takeIf { it.kind == Kind.GUEST }?.token

    /** Must be called with [mutex] held. */
    private suspend fun signInAsGuest(): Session {
        val session = authApi.anonymous(store.deviceId()).toSession()
        store.saveSession(session)
        return session
    }

    private fun AuthResponse.toSession() = Session(token, userId, kind, plan)
}
