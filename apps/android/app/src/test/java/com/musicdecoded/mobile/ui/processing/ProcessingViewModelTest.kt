package com.musicdecoded.mobile.ui.processing

import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.network.JobStatus
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
class ProcessingViewModelTest {
    private val item = MutableStateFlow<LibraryItem?>(null)
    private val jobs = MutableStateFlow<Map<String, Job>>(emptyMap())
    private val retried = mutableListOf<String>()

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    private fun row(status: LibraryStatus, error: String? = null) = LibraryItem(
        trackId = "t1",
        title = "Song",
        artist = "Artist",
        source = Source.FILE,
        createdAt = Instant.EPOCH,
        updatedAt = Instant.EPOCH,
        status = status,
        error = error,
    )

    private fun job(stage: String?, progress: Int, status: JobStatus = JobStatus.PROCESSING) = Job("j1", "t1", status, stage, progress)

    private fun runVm(block: suspend (ProcessingViewModel) -> Unit) = runTest(UnconfinedTestDispatcher()) {
        val vm = ProcessingViewModel("t1", item, jobs) { retried += it }
        backgroundScope.launch { vm.state.collect {} }
        block(vm)
    }

    /** DESIGN.md §3.5, row by row. */
    @Test fun stageToStepMatchesTheDesignTable() {
        val table = mapOf(
            "fetching_media" to ProcessingStep.PREPARING,
            "normalizing_audio" to ProcessingStep.PREPARING,
            "detecting_beats" to ProcessingStep.BEATS,
            "detecting_chords" to ProcessingStep.HARMONY,
            "detecting_key" to ProcessingStep.HARMONY,
            "rendering_spectrogram" to ProcessingStep.TIMELINE,
            "saving_results" to ProcessingStep.SAVING,
        )
        table.forEach { (stage, step) -> assertEquals(stage, step, ProcessingStep.forStage(stage)) }
        assertEquals(table.keys, ProcessingStep.entries.flatMap { it.stages }.toSet())
        assertEquals(
            listOf("Preparing audio", "Detecting beats & tempo", "Analyzing harmony", "Generating chord timeline", "Saving transcription"),
            ProcessingStep.entries.map { it.label },
        )
        assertNull(ProcessingStep.forStage(null))
        assertNull(ProcessingStep.forStage("brand_new_stage"))
    }

    @Test fun loadingUntilTheRowExists() = runVm { vm ->
        assertEquals(ProcessingPhase.LOADING, vm.state.value.phase)
    }

    @Test fun runningFollowsStageAndProgress() = runVm { vm ->
        item.value = row(LibraryStatus.ANALYZING)
        assertEquals(ProcessingPhase.RUNNING, vm.state.value.phase)
        assertEquals(ProcessingStep.PREPARING, vm.state.value.step) // queued, no stage yet
        assertEquals(0, vm.state.value.progress)

        jobs.value = mapOf("t1" to job("detecting_chords", 55))
        val state = vm.state.value
        assertEquals(ProcessingStep.HARMONY, state.step)
        assertEquals(55, state.progress)
        assertTrue(state.isDone(ProcessingStep.PREPARING))
        assertTrue(state.isDone(ProcessingStep.BEATS))
        assertTrue(state.isActive(ProcessingStep.HARMONY))
        assertFalse(state.isDone(ProcessingStep.TIMELINE))
        assertEquals("Song", state.title)
    }

    @Test fun unknownStageKeepsTheStepAndProgressNeverGoesBack() = runVm { vm ->
        item.value = row(LibraryStatus.ANALYZING)
        jobs.value = mapOf("t1" to job("rendering_spectrogram", 80))
        jobs.value = mapOf("t1" to job("something_new", 70))
        assertEquals(ProcessingStep.TIMELINE, vm.state.value.step)
        assertEquals(80, vm.state.value.progress)
    }

    @Test fun completeJobIsSavingUntilTheRowIsReady() = runVm { vm ->
        item.value = row(LibraryStatus.ANALYZING)
        jobs.value = mapOf("t1" to job(null, 100, JobStatus.COMPLETE))
        assertEquals(ProcessingStep.SAVING, vm.state.value.step)
        assertEquals(ProcessingPhase.RUNNING, vm.state.value.phase)

        jobs.value = emptyMap()
        item.value = row(LibraryStatus.READY)
        val state = vm.state.value
        assertEquals(ProcessingPhase.COMPLETE, state.phase)
        assertEquals(100, state.progress)
        assertTrue(ProcessingStep.entries.all(state::isDone))
    }

    @Test fun failedShowsTheErrorAndRetryCallsThrough() = runVm { vm ->
        item.value = row(LibraryStatus.FAILED, error = "We couldn't decode this audio. Try again.")
        assertEquals(ProcessingPhase.FAILED, vm.state.value.phase)
        assertEquals("We couldn't decode this audio. Try again.", vm.state.value.error)
        vm.retry()
        assertEquals(listOf("t1"), retried)

        item.value = row(LibraryStatus.ANALYZING)
        assertEquals(ProcessingPhase.RUNNING, vm.state.value.phase)
        assertNull(vm.state.value.error)
    }

    @Test fun deletedRowIsLoading() = runVm { vm ->
        item.value = row(LibraryStatus.ANALYZING).copy(deleted = true)
        assertEquals(ProcessingPhase.LOADING, vm.state.value.phase)
    }
}
