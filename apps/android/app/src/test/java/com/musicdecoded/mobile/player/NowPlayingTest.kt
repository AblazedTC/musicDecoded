package com.musicdecoded.mobile.player

import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.core.SyntheticAnalysis
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** The app-level session behind the mini player, with a fake-clock [SilentPlaybackEngine]. */
@OptIn(ExperimentalCoroutinesApi::class)
class NowPlayingTest {
    private var now = 0L
    private val released = mutableListOf<PlaybackEngine>()
    private val nowPlaying = NowPlaying()
    private val track = PlayerTrack(SyntheticAnalysis.make(), "Song", "Artist", favorite = false, audio = null)

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    private fun advance(seconds: Double) {
        now += (seconds * 1e9).toLong()
    }

    private fun engine(t: PlayerTrack): PlaybackEngine = object : PlaybackEngine by SilentPlaybackEngine(t.analysis.meta.durationSec, { now }) {
        override fun release() {
            released += this
        }
    }

    /** A player ViewModel owned by [store], built the way the app's factory does (engine from [NowPlaying]). */
    private fun viewModel(store: ViewModelStore = ViewModelStore(), trackId: String = "t1"): PlayerViewModel {
        val factory = viewModelFactory {
            initializer {
                PlayerViewModel(
                    loadTrack = { track },
                    voicings = null,
                    createEngine = { nowPlaying.engineFor(trackId, it, ::engine) },
                    ownsEngine = false,
                )
            }
        }
        return ViewModelProvider.create(store, factory)[PlayerViewModel::class]
    }

    @Test fun playbackOutlivesThePlayerAndReopeningReusesTheEngine() {
        val playerEntry = ViewModelStore()
        val first = viewModel(playerEntry)
        first.togglePlay()
        first.setSpeed(0.75f)
        advance(2.0)
        assertEquals("Song", nowPlaying.state.value?.title)

        // Leave the player: the engine isn't released and the mini player keeps the clock.
        playerEntry.clear() // what popping the player's back stack entry does
        assertTrue(released.isEmpty())
        nowPlaying.tick()
        assertTrue(nowPlaying.state.value!!.isPlaying)
        assertEquals(1.5, nowPlaying.position.value, 1e-6)

        // Reopen: same engine, state picked up (playing, speed, position).
        val second = viewModel()
        assertTrue(second.state.value.isPlaying)
        assertEquals(0.75f, second.state.value.speed)
        assertEquals(1.5, second.position.value, 1e-6)
    }

    @Test fun anotherTrackReplacesAndReleasesTheOldEngine() {
        val engineA = nowPlaying.engineFor("a", track, ::engine)
        assertSame(engineA, nowPlaying.engineFor("a", track, ::engine))
        val engineB = nowPlaying.engineFor("b", track, ::engine)
        assertNotSame(engineA, engineB)
        assertEquals(listOf(engineA), released)
        assertEquals("b", nowPlaying.state.value?.trackId)
    }

    @Test fun miniPlayerControlsAndEnd() {
        nowPlaying.engineFor("a", track, ::engine)
        nowPlaying.togglePlay()
        assertTrue(nowPlaying.state.value!!.isPlaying)
        advance(100.0) // past the 10 s synthetic track
        nowPlaying.tick()
        assertFalse(nowPlaying.state.value!!.isPlaying)
        assertEquals(10.0, nowPlaying.position.value, 1e-6)

        nowPlaying.togglePlay() // at the end: restarts from the top
        assertTrue(nowPlaying.state.value!!.isPlaying)
        assertEquals(0.0, nowPlaying.position.value, 1e-6)

        nowPlaying.pause()
        assertFalse(nowPlaying.state.value!!.isPlaying)
        nowPlaying.dismiss()
        assertNull(nowPlaying.state.value)
        assertEquals(1, released.size)
    }
}
