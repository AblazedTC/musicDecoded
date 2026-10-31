package com.musicdecoded.mobile

import android.app.Application
import android.util.Log
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.auth.TokenStore
import com.musicdecoded.mobile.billing.BillingService
import com.musicdecoded.mobile.billing.RevenueCatBillingService
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.library.AnalysisStore
import com.musicdecoded.mobile.library.DataStoreSyncState
import com.musicdecoded.mobile.library.SyncRepository
import com.musicdecoded.mobile.library.SyncWorker
import com.musicdecoded.mobile.network.SyncApiService
import com.musicdecoded.mobile.settings.SettingsStore
import com.musicdecoded.mobile.library.LibraryDatabase
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.library.TranscriptionWorker
import com.musicdecoded.mobile.network.TranscriptionApiService
import com.musicdecoded.mobile.player.NowPlaying
import java.io.File
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.map
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

    /** Play Billing via RevenueCat (T-602a); unavailable when the build has no key. */
    val billingService: BillingService by lazy {
        RevenueCatBillingService(this, BuildConfig.REVENUECAT_API_KEY, debugLogs = BuildConfig.DEBUG)
    }

    val libraryDatabase by lazy { LibraryDatabase.create(this) }
    val analysisStore by lazy { AnalysisStore(File(filesDir, "analysis")) }
    val transcriptionRepository by lazy {
        TranscriptionRepository(
            TranscriptionApiService(httpClient, BuildConfig.API_BASE_URL),
            authRepository,
            libraryDatabase.libraryDao(),
            analysisStore,
            filesDir,
            TranscriptionWorker.Scheduler(this),
            onJobComplete = ::requestSync,
        )
    }

    val settingsStore by lazy { SettingsStore(this) }
    val syncRepository by lazy {
        SyncRepository(
            SyncApiService(httpClient, BuildConfig.API_BASE_URL),
            TranscriptionApiService(httpClient, BuildConfig.API_BASE_URL),
            authRepository,
            libraryDatabase.libraryDao(),
            analysisStore,
            filesDir,
            DataStoreSyncState(this),
        )
    }

    /** Background library sync (accounts only; a no-op for guests). PLAN.md §5.2 triggers call this. */
    fun requestSync() = SyncWorker.schedule(this)

    /** The track loaded in the player; outlives the player screen (mini player, T-211). */
    val nowPlaying by lazy { NowPlaying() }

    override fun onCreate() {
        super.onCreate()
        // Guest session + /config at launch. Offline is fine: the cached config is used and the
        // session is created on the next API call.
        appScope.launch {
            runCatching { authRepository.start() }
                .onFailure { Log.w("MusicDecoded", "Session start failed; continuing offline", it) }
        }
        // The store identity follows the session: appUserID = our userId for accounts (PLAN.md §9).
        appScope.launch {
            authRepository.session.filterNotNull().map { it.takeIf { s -> s.kind == Kind.ACCOUNT }?.userId }
                .distinctUntilChanged()
                .collect { userId ->
                    if (!billingService.isAvailable) return@collect
                    runCatching { if (userId != null) billingService.logIn(userId) else billingService.logOut() }
                        .onFailure { Log.w("MusicDecoded", "Billing identity update failed", it) }
                }
        }
        // Jobs that were still running when the process died resume polling (WorkManager keeps
        // already-scheduled work, so this only fills gaps).
        appScope.launch { transcriptionRepository.resumePending() }
    }
}
