package com.musicdecoded.mobile.ui.library

import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryCounts
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.network.JobStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.time.Instant

@OptIn(ExperimentalCoroutinesApi::class)
class LibraryViewModelTest {
    private val items = MutableStateFlow<List<LibraryItem>>(emptyList())
    private val counts = MutableStateFlow(LibraryCounts(0, 0))
    private val jobs = MutableStateFlow<Map<String, Job>>(emptyMap())
    private val queries = mutableListOf<LibraryQuery>()
    private val actions = SongActions(delete = {}, setFavorite = { _, _ -> }, reanalyze = {})

    /** Fake DAO search: records the query and returns [items] (the database does the real filtering). */
    private val search: (LibraryQuery) -> Flow<List<LibraryItem>> = { query ->
        queries += query
        items.map { it }
    }

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    private fun item(id: String, status: LibraryStatus = LibraryStatus.READY) = LibraryItem(
        trackId = id,
        title = "Song $id",
        source = Source.FILE,
        createdAt = Instant.EPOCH,
        updatedAt = Instant.EPOCH,
        status = status,
    )

    private fun runVm(block: suspend (LibraryViewModel) -> Unit) = runTest(UnconfinedTestDispatcher()) {
        val vm = LibraryViewModel(search, counts, jobs, actions)
        backgroundScope.launch { vm.state.collect {} }
        block(vm)
    }

    @Test fun startsWithAllSongsAndNoText() = runVm { vm ->
        assertEquals(LibraryQuery("", favoritesOnly = false, offlineOnly = false, source = null), queries.single())
        assertTrue(vm.state.value.libraryEmpty)
    }

    @Test fun searchTextIsTrimmedMappedToStoredAccidentalsAndEscaped() = runVm { vm ->
        vm.setQuery("  F♯ min ")
        assertEquals("F# min", queries.last().pattern)
        assertEquals("  F♯ min ", vm.searchText.value)
        vm.setQuery("B♭")
        assertEquals("Bb", queries.last().pattern)
        vm.setQuery("100%_!")
        assertEquals("100!%!_!!", queries.last().pattern)
    }

    @Test fun filtersMapToQueryFlags() = runVm { vm ->
        vm.setFilter(LibraryFilter.FAVORITES)
        assertEquals(LibraryQuery("", favoritesOnly = true, offlineOnly = false, source = null), queries.last())
        vm.setFilter(LibraryFilter.OFFLINE)
        assertEquals(LibraryQuery("", favoritesOnly = false, offlineOnly = true, source = null), queries.last())
        vm.setFilter(LibraryFilter.RECORDINGS)
        assertEquals(LibraryQuery("", favoritesOnly = false, offlineOnly = false, source = "RECORDING"), queries.last())
        assertEquals(LibraryFilter.RECORDINGS, vm.state.value.filter)
    }

    @Test fun sameQueryDoesNotSearchAgain() = runVm { vm ->
        vm.setQuery("a ")
        vm.setQuery("a")
        assertEquals(2, queries.size)
    }

    @Test fun rowsCountsAndProgress() = runVm { vm ->
        items.value = listOf(item("b", LibraryStatus.ANALYZING), item("a"))
        counts.value = LibraryCounts(decoded = 1, offline = 2)
        jobs.value = mapOf("b" to Job("j", "b", JobStatus.PROCESSING, "detecting_chords", 60))
        val state = vm.state.value
        assertEquals(listOf("b", "a"), state.rows.map { it.trackId })
        assertEquals(60, state.rows.first().progress)
        assertEquals(LibraryCounts(1, 2), state.counts)
        assertFalse(state.libraryEmpty)
    }

    @Test fun noMatchesIsNotAnEmptyLibrary() = runVm { vm ->
        vm.setQuery("zzz")
        assertTrue(vm.state.value.rows.isEmpty())
        assertFalse(vm.state.value.libraryEmpty)
    }
}
