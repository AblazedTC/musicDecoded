package com.musicdecoded.mobile.ui.home

import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.network.JobStatus
import com.musicdecoded.mobile.network.Plan
import com.musicdecoded.mobile.ui.components.SongRowStatus
import com.musicdecoded.mobile.ui.library.SongActions
import com.musicdecoded.mobile.ui.library.songMeta
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.time.Instant

@OptIn(ExperimentalCoroutinesApi::class)
class HomeViewModelTest {
    private val items = MutableStateFlow<List<LibraryItem>>(emptyList())
    private val jobs = MutableStateFlow<Map<String, Job>>(emptyMap())
    private val config = MutableStateFlow<ClientConfig?>(null)
    private val calls = mutableListOf<String>()
    private var reanalyzeError: Exception? = null

    private val actions = SongActions(
        delete = { calls += "delete:$it" },
        setFavorite = { id, fav -> calls += "favorite:$id:$fav" },
        reanalyze = { id -> reanalyzeError?.let { throw it }; calls += "reanalyze:$id" },
    )

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    private fun item(id: String, status: LibraryStatus = LibraryStatus.READY, local: Boolean = true) = LibraryItem(
        trackId = id,
        title = "Song $id",
        artist = "Artist",
        source = Source.FILE,
        durationSec = 263.4,
        key = "F# minor",
        bpm = 136.1,
        createdAt = Instant.EPOCH,
        updatedAt = Instant.EPOCH,
        status = status,
        localAudioPath = if (local) "audio/$id.m4a" else null,
    )

    private fun runVm(block: suspend (HomeViewModel) -> Unit) = runTest(UnconfinedTestDispatcher()) {
        val vm = HomeViewModel(items, jobs, config, actions)
        backgroundScope.launch { vm.state.collect {} }
        block(vm)
    }

    @Test fun emptyHistoryIsLoadedAndEmpty() = runVm { vm ->
        assertFalse(vm.state.value.loading)
        assertTrue(vm.state.value.recent.isEmpty())
    }

    @Test fun showsAtMostThreeRowsInOrder() = runVm { vm ->
        items.value = listOf(item("a"), item("b"), item("c"), item("d"))
        assertEquals(listOf("a", "b", "c"), vm.state.value.recent.map { it.trackId })
    }

    @Test fun rowShowsMetaOfflineAndStatus() = runVm { vm ->
        items.value = listOf(item("a"), item("b", LibraryStatus.FAILED, local = false))
        val (ready, failed) = vm.state.value.recent
        assertEquals("4:23 · F♯ minor · 136 BPM", ready.meta)
        assertTrue(ready.offline)
        assertEquals(SongRowStatus.READY, ready.status)
        assertEquals(SongRowStatus.FAILED, failed.status)
        assertFalse(failed.offline)
    }

    @Test fun analyzingRowTakesProgressFromJobs() = runVm { vm ->
        items.value = listOf(item("a", LibraryStatus.ANALYZING))
        assertNull(vm.state.value.recent.single().progress)
        jobs.value = mapOf("a" to Job("j1", "a", JobStatus.PROCESSING, "detecting_beats", 42))
        assertEquals(SongRowStatus.ANALYZING, vm.state.value.recent.single().status)
        assertEquals(42, vm.state.value.recent.single().progress)
    }

    @Test fun soundcloudFollowsConfig() = runVm { vm ->
        assertFalse(vm.state.value.soundcloudEnabled)
        config.value = ClientConfig(false, 1, soundcloudEnabled = true, plan = Plan.FREE, limits = ClientConfig.Limits(null, 0))
        assertTrue(vm.state.value.soundcloudEnabled)
    }

    @Test fun menuActionsCallThrough() = runVm { vm ->
        vm.delete("a")
        vm.setFavorite("a", true)
        vm.reanalyze("a")
        assertEquals(listOf("delete:a", "favorite:a:true", "reanalyze:a"), calls)
    }

    @Test fun failedReanalyzeShowsTheMessageOnce() = runVm { vm ->
        reanalyzeError = TranscriptionException("Nope")
        vm.reanalyze("a")
        assertEquals("Nope", vm.message.value)
        vm.messageShown()
        assertNull(vm.message.value)
    }

    @Test fun metaSkipsUnknownParts() {
        assertEquals("", songMeta(null, null, null))
        assertEquals("B♭ major", songMeta(null, "Bb major", null))
    }
}
