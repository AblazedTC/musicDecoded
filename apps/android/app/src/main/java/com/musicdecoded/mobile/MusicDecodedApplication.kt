package com.musicdecoded.mobile

import android.app.Application
import android.util.Log
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.auth.TokenStore
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/** Manual DI root (apps/android/AGENTS.md). */
class MusicDecodedApplication : Application() {
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    val httpClient by lazy { createHttpClient(Android.create(), debugLogging = BuildConfig.DEBUG) }
    val authRepository by lazy {
        AuthRepository(
            AuthApiService(httpClient, BuildConfig.API_BASE_URL),
            ConfigApiService(httpClient, BuildConfig.API_BASE_URL),
            TokenStore(this),
        )
    }

    override fun onCreate() {
        super.onCreate()
        // Guest session + /config at launch. Offline is fine: the cached config is used and the
        // session is created on the next API call.
        appScope.launch {
            runCatching { authRepository.start() }
                .onFailure { Log.w("MusicDecoded", "Session start failed; continuing offline", it) }
        }
    }
}
