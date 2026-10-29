package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Contracts
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.core.SyntheticAnalysis
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** Player state logic against [SilentPlaybackEngine] driven by a fake clock (no ExoPlayer). */
@OptIn(ExperimentalCoroutinesApi::class)
class PlayerViewModelTest {
    /** Fake monotonic clock, in nanoseconds. */
    private var now = 0L
    private var engine: SilentPlaybackEngine? = null
    private val savedFavorites = mutableListOf<Boolean>()

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    private fun advance(seconds: Double) {
        now += (seconds * 1e9).toLong()
    }

    /** Synthetic: A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10; beats every 1 s, downbeats 0, 4, 8. */
    private fun viewModel(analysis: Analysis = SyntheticAnalysis.make(), track: Boolean = true) = PlayerViewModel(
        loadTrack = { if (track) PlayerTrack(analysis, "Song", "Artist", favorite = false, audio = null, videoId = analysis.meta.sourceRef) else null },
        voicings = Contracts.voicings(),
        createEngine = { SilentPlaybackEngine(it.analysis.meta.durationSec) { now }.also { engine = it } },
        saveFavorite = { savedFavorites += it },
    )

    @Test fun loadsTheTrack() {
        val vm = viewModel()
        assertFalse(vm.state.value.loading)
        assertEquals("Song", vm.state.value.track?.title)
        assertEquals(0.0, vm.position.value, 0.0)
    }

    @Test fun missingTrackStopsLoadingWithoutATrack() {
        val vm = viewModel(track = false)
        assertFalse(vm.state.value.loading)
        assertNull(vm.state.value.track)
        vm.togglePlay() // no engine: no crash
        assertFalse(vm.state.value.isPlaying)
    }

    @Test fun engineClockDrivesThePosition() {
        val vm = viewModel()
        vm.togglePlay()
        assertTrue(vm.state.value.isPlaying)
        advance(1.5)
        vm.tick()
        assertEquals(1.5, vm.position.value, 1e-6)
        vm.togglePlay()
        advance(3.0)
        vm.tick()
        assertFalse(vm.state.value.isPlaying)
        assertEquals(1.5, vm.position.value, 1e-6)
    }

    @Test fun speedScalesTheClockAndSnapsToSteps() {
        val vm = viewModel()
        vm.setSpeed(0.77f)
        assertEquals(0.75f, vm.state.value.speed)
        vm.togglePlay()
        advance(2.0)
        vm.tick()
        assertEquals(1.5, vm.position.value, 1e-6)
        vm.setSpeed(3f)
        assertEquals(1.25f, vm.state.value.speed)
        vm.setSpeed(0.1f)
        assertEquals(0.5f, vm.state.value.speed)
    }

    @Test fun speedTapTogglesBetweenNormalAndPractice() {
        val vm = viewModel()
        vm.toggleSpeed()
        assertEquals(0.75f, vm.state.value.speed)
        vm.toggleSpeed()
        assertEquals(1f, vm.state.value.speed)
        vm.setSpeed(1.1f) // any non-1× speed taps back to 1×
        vm.toggleSpeed()
        assertEquals(1f, vm.state.value.speed)
    }

    @Test fun speedStepsCover050to125() {
        assertEquals(16, PlayerViewModel.SPEED_STEPS.size)
        assertEquals(0.5f, PlayerViewModel.SPEED_STEPS.first())
        assertEquals(1.25f, PlayerViewModel.SPEED_STEPS.last())
    }

    @Test fun seekIsClampedToTheTrack() {
        val vm = viewModel()
        vm.seekTo(-3.0)
        assertEquals(0.0, vm.position.value, 0.0)
        vm.seekTo(99.0)
        assertEquals(10.0, vm.position.value, 0.0)
        assertEquals(10.0, engine!!.position(), 0.0)
    }

    @Test fun nextAndPreviousChordSkipNoChord() {
        val vm = viewModel()
        vm.seekTo(0.5)
        vm.nextChord()
        assertEquals(5.0, vm.position.value, 0.0) // skips the N runs
        vm.seekTo(6.5) // > 1 s into D:maj → restart it
        vm.previousChord()
        assertEquals(5.0, vm.position.value, 0.0)
        vm.previousChord() // at its start → previous real chord
        assertEquals(0.0, vm.position.value, 0.0)
    }

    @Test fun loopButtonLoopsTheCurrentBar() {
        val vm = viewModel()
        vm.seekTo(5.5)
        vm.toggleLoop()
        assertEquals(4.0..<8.0, vm.state.value.loop)
        vm.togglePlay()
        advance(2.6) // 5.5 + 2.6 = 8.1 → wraps to the bar start
        vm.tick()
        assertEquals(4.0, vm.position.value, 1e-6)
        advance(1.0)
        vm.tick()
        assertEquals(5.0, vm.position.value, 1e-6)
        vm.toggleLoop()
        assertNull(vm.state.value.loop)
    }

    @Test fun customLoopJumpsInWhenThePlayheadIsOutside() {
        val vm = viewModel()
        vm.seekTo(0.5)
        vm.setLoop(2.0..<4.0)
        assertEquals(2.0, vm.position.value, 0.0)
        vm.clearLoop()
        assertNull(vm.state.value.loop)
        vm.seekTo(3.0)
        vm.setLoop(2.0..<6.0) // already inside: stays
        assertEquals(3.0, vm.position.value, 0.0)
    }

    @Test fun stopsAtTheEndAndRestartsFromTheTop() {
        val vm = viewModel()
        vm.seekTo(9.0)
        vm.togglePlay()
        advance(5.0)
        vm.tick()
        assertFalse(vm.state.value.isPlaying)
        assertEquals(10.0, vm.position.value, 1e-6)
        vm.togglePlay()
        assertTrue(vm.state.value.isPlaying)
        assertEquals(0.0, vm.position.value, 1e-6)
    }

    @Test fun pauseOnlyPauses() {
        val vm = viewModel()
        vm.pause()
        assertFalse(vm.state.value.isPlaying)
        vm.togglePlay()
        vm.pause()
        assertFalse(vm.state.value.isPlaying)
    }

    @Test fun transposeIsClampedAndChangesNamesAndNotes() {
        val vm = viewModel()
        vm.setTranspose(9)
        assertEquals(6, vm.state.value.transpose)
        vm.setTranspose(-9)
        assertEquals(-6, vm.state.value.transpose)
        vm.setTranspose(3)
        val chords = vm.state.value.chords
        assertEquals("C", chords.name("A:maj"))
        assertEquals(Contracts.voicings().notes("C:maj"), chords.notes("A:maj"))
        assertEquals("C · E · G", chords.noteNames("A:maj"))
    }

    @Test fun basicChordsSimplifyNamesAndNotes() {
        val vm = viewModel()
        assertEquals("F♯m7", vm.state.value.chords.name("F#:min7"))
        vm.toggleChordsMode()
        assertEquals(ChordsMode.BASIC, vm.state.value.chordsMode)
        val chords = vm.state.value.chords
        assertEquals("F♯m", chords.name("F#:min7"))
        assertEquals(Contracts.voicings().notes("F#:min"), chords.notes("F#:min7"))
        assertNull(chords.bassNote("D:maj/3"))
    }

    @Test fun slashBassIsTheLowestNote() {
        val chords = viewModel().state.value.chords
        val notes = chords.notes("D:maj/3")
        assertEquals(notes.first(), chords.bassNote("D:maj/3"))
        assertTrue(notes.first() < notes[1])
        assertNull(chords.bassNote("D:maj"))
    }

    @Test fun favoriteTogglesAndSaves() {
        val vm = viewModel()
        vm.toggleFavorite()
        assertTrue(vm.state.value.favorite)
        vm.toggleFavorite()
        assertEquals(listOf(true, false), savedFavorites)
    }

    @Test fun videoOnlyForYoutubeSources() {
        val vm = viewModel()
        vm.selectVisualizer(Visualizer.VIDEO)
        assertEquals(Visualizer.PIANO, vm.state.value.visualizer)

        val youtube = SyntheticAnalysis.make().let { it.copy(meta = it.meta.copy(source = Source.YOUTUBE, sourceRef = "abc")) }
        val ytVm = viewModel(youtube)
        assertTrue(ytVm.state.value.videoAvailable)
        ytVm.selectVisualizer(Visualizer.VIDEO)
        assertEquals(Visualizer.VIDEO, ytVm.state.value.visualizer)
    }

    @Test fun noteNamesToggle() {
        val vm = viewModel()
        assertTrue(vm.state.value.showNoteNames)
        vm.toggleNoteNames()
        assertFalse(vm.state.value.showNoteNames)
    }

    @Test fun formatsChipValues() {
        assertEquals("+2", formatTranspose(2))
        assertEquals("−3", formatTranspose(-3))
        assertEquals("0", formatTranspose(0))
        assertEquals("0.75×", formatSpeed(0.75f))
    }
}
