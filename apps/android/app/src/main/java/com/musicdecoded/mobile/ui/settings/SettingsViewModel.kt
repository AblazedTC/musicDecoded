package com.musicdecoded.mobile.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.auth.Session
import com.musicdecoded.mobile.library.LibraryCounts
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.settings.PlayerPreferences
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

data class SettingsUiState(
    val session: Session? = null,
    val preferences: PlayerPreferences = PlayerPreferences(),
    /** Bytes of audio + analysis files on this device. */
    val storageBytes: Long = 0,
    /** Songs whose audio is on this device. */
    val offlineFiles: Int = 0,
) {
    val isAccount: Boolean get() = session?.kind == Kind.ACCOUNT
}

/** DESIGN.md §3.9 (v1 rows only). */
class SettingsViewModel(
    session: StateFlow<Session?>,
    preferences: Flow<PlayerPreferences>,
    counts: Flow<LibraryCounts>,
    private val storageBytes: suspend () -> Long,
    private val refreshSession: suspend () -> Unit,
    private val updatePreferences: suspend ((PlayerPreferences) -> PlayerPreferences) -> Unit,
    private val logOut: suspend () -> Unit,
) : ViewModel() {
    /** Storage is re-measured whenever the library changes. */
    private val storage = counts.map { it to storageBytes() }.flowOn(Dispatchers.IO)

    val state: StateFlow<SettingsUiState> = combine(session, preferences, storage) { s, p, (c, bytes) ->
        SettingsUiState(session = s, preferences = p, storageBytes = bytes, offlineFiles = c.offline)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SettingsUiState())

    init {
        viewModelScope.launch { runCatching { refreshSession() } }
    }

    fun update(transform: (PlayerPreferences) -> PlayerPreferences) {
        viewModelScope.launch { updatePreferences(transform) }
    }

    /** Back to this device's guest session; the local library stays (T-403a). */
    fun logout() {
        viewModelScope.launch { runCatching { logOut() } }
    }

    companion object {
        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                SettingsViewModel(
                    session = app.authRepository.session,
                    preferences = app.settingsStore.preferences,
                    counts = app.libraryDatabase.libraryDao().counts(),
                    storageBytes = { directorySize(File(app.filesDir, "audio")) + directorySize(File(app.filesDir, "analysis")) },
                    refreshSession = { app.authRepository.currentSession() },
                    updatePreferences = app.settingsStore::update,
                    logOut = { app.authRepository.logout() },
                )
            }
        }

        fun directorySize(dir: File): Long = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }
}
