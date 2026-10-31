package com.musicdecoded.mobile.ui.home

import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.ui.library.SongActions
import com.musicdecoded.mobile.ui.library.SongListViewModel
import com.musicdecoded.mobile.ui.library.SongRowState
import com.musicdecoded.mobile.ui.library.toRowState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn

data class HomeUiState(
    val loading: Boolean = true,
    /** Up to [HomeViewModel.RECENT_COUNT] newest songs. Empty = show the empty-history prompt. */
    val recent: List<SongRowState> = emptyList(),
    /** The SoundCloud card is hidden unless the server enables it. */
    val soundcloudEnabled: Boolean = false,
)

/** DESIGN.md §3.1. [recent] is the DAO's newest-first rows (already limited), [jobs] the running jobs. */
class HomeViewModel(
    recent: Flow<List<LibraryItem>>,
    jobs: StateFlow<Map<String, Job>>,
    config: StateFlow<ClientConfig?>,
    actions: SongActions,
) : SongListViewModel(actions) {
    val state: StateFlow<HomeUiState> = combine(recent, jobs, config) { items, running, cfg ->
        HomeUiState(
            loading = false,
            recent = items.take(RECENT_COUNT).map { it.toRowState(running[it.trackId]) },
            soundcloudEnabled = cfg?.soundcloudEnabled == true,
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), HomeUiState())

    companion object {
        const val RECENT_COUNT = 3

        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                HomeViewModel(
                    recent = app.libraryDatabase.libraryDao().recent(RECENT_COUNT),
                    jobs = app.transcriptionRepository.jobs,
                    config = app.authRepository.config,
                    actions = SongActions.from(app),
                )
            }
        }
    }
}
