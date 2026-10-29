package com.musicdecoded.mobile.ui.library

import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryCounts
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.likePattern
import com.musicdecoded.mobile.network.Job
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn

enum class LibraryFilter(val label: String) {
    ALL("All songs"),
    FAVORITES("Favorites"),
    OFFLINE("Offline"),
    RECORDINGS("Recordings"),
}

/** The arguments of [com.musicdecoded.mobile.library.LibraryDao.search] for a search text + filter. */
data class LibraryQuery(val pattern: String, val favoritesOnly: Boolean, val offlineOnly: Boolean, val source: String?) {
    companion object {
        /** Trims the text, maps ♯/♭ to the `#`/`b` stored in keys, and escapes it for `LIKE`. */
        fun of(text: String, filter: LibraryFilter) = LibraryQuery(
            pattern = likePattern(text.trim().replace("♯", "#").replace("♭", "b")),
            favoritesOnly = filter == LibraryFilter.FAVORITES,
            offlineOnly = filter == LibraryFilter.OFFLINE,
            source = Source.RECORDING.name.takeIf { filter == LibraryFilter.RECORDINGS },
        )
    }
}

data class LibraryUiState(
    val loading: Boolean = true,
    val query: String = "",
    val filter: LibraryFilter = LibraryFilter.ALL,
    val rows: List<SongRowState> = emptyList(),
    val counts: LibraryCounts = LibraryCounts(0, 0),
) {
    /** Nothing decoded or offline at all: show the "decode a song from Home" prompt, not "no matches". */
    val libraryEmpty: Boolean get() = !loading && rows.isEmpty() && query.isBlank() && filter == LibraryFilter.ALL
}

/** DESIGN.md §3.2. Searching and filtering run in the database ([search]), not over the whole list. */
@OptIn(ExperimentalCoroutinesApi::class)
class LibraryViewModel(
    search: (LibraryQuery) -> Flow<List<LibraryItem>>,
    counts: Flow<LibraryCounts>,
    jobs: StateFlow<Map<String, Job>>,
    actions: SongActions,
) : SongListViewModel(actions) {
    private val query = MutableStateFlow("")

    /** The search text, bound straight to the text field (so typing never waits on the database). */
    val searchText: StateFlow<String> = query.asStateFlow()
    private val filter = MutableStateFlow(LibraryFilter.ALL)

    private val results: Flow<List<LibraryItem>> = combine(query, filter, LibraryQuery::of)
        .distinctUntilChanged()
        .flatMapLatest(search)

    val state: StateFlow<LibraryUiState> = combine(query, filter, results, counts, jobs) { text, f, items, c, running ->
        LibraryUiState(
            loading = false,
            query = text,
            filter = f,
            rows = items.map { it.toRowState(running[it.trackId]) },
            counts = c,
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), LibraryUiState())

    fun setQuery(text: String) {
        query.value = text
    }

    fun setFilter(value: LibraryFilter) {
        filter.value = value
    }

    companion object {
        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val dao = app.libraryDatabase.libraryDao()
                LibraryViewModel(
                    search = { dao.search(it.pattern, it.favoritesOnly, it.offlineOnly, it.source) },
                    counts = dao.counts(),
                    jobs = app.transcriptionRepository.jobs,
                    actions = SongActions.from(app),
                )
            }
        }
    }
}
