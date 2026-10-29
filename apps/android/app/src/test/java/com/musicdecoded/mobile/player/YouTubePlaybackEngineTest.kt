package com.musicdecoded.mobile.player

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

/** Records commands; the test plays the page's part by calling the listener. */
private class FakeBridge : YouTubePlayerBridge {
    override var listener: YouTubePlayerBridge.Listener? = null
    val commands = mutableListOf<String>()
    override fun play() { commands += "play" }
    override fun pause() { commands += "pause" }
    override fun seekTo(seconds: Double) { commands += "seek $seconds" }
    override fun setPlaybackRate(rate: Float) { commands += "rate $rate" }
    override fun mute() { commands += "mute" }

    fun state(state: YouTubePlayerState) = listener!!.onStateChange(state)
    fun time(seconds: Double) = listener!!.onTime(seconds)
}

@OptIn(ExperimentalCoroutinesApi::class)
class YouTubePlaybackEngineTest {
    private var now = 0L
    private fun advance(seconds: Double) { now += (seconds * 1e9).toLong() }

    private fun engine(muted: Boolean = false) = YouTubePlaybackEngine(durationSec = 10.0, muted = muted) { now }

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    @Test fun cannotPlayWithoutAVideoOnScreen() {
        val engine = engine()
        engine.play()
        assertFalse(engine.isPlaying)
        engine.seekTo(3.0)
        assertEquals(3.0, engine.position(), 0.0)
    }

    @Test fun playCuesAnIdleVideoAtOurPositionThenPlays() {
        val engine = engine()
        val bridge = FakeBridge()
        engine.attach(bridge)
        engine.seekTo(4.5)
        engine.play()
        assertTrue(engine.isPlaying)
        assertEquals(listOf("seek 4.5", "seek 4.5", "play"), bridge.commands)
    }

    @Test fun stateMapping() {
        val engine = engine()
        val bridge = FakeBridge()
        var notified = 0
        engine.onPlayStateChange = { notified++ }
        engine.attach(bridge)

        // A tap on the embed's own play button.
        bridge.state(YouTubePlayerState.BUFFERING)
        assertFalse(engine.isPlaying)
        bridge.state(YouTubePlayerState.PLAYING)
        assertTrue(engine.isPlaying)
        assertEquals(1, notified)
        // Buffering mid-play keeps the intent.
        bridge.state(YouTubePlayerState.BUFFERING)
        assertTrue(engine.isPlaying)
        assertTrue(engine.buffering)
        bridge.state(YouTubePlayerState.PLAYING)
        bridge.state(YouTubePlayerState.PAUSED)
        assertFalse(engine.isPlaying)
        assertEquals(2, notified)
        bridge.state(YouTubePlayerState.PLAYING)
        bridge.state(YouTubePlayerState.ENDED)
        assertFalse(engine.isPlaying)
        assertEquals(10.0, engine.position(), 0.0) // ended reads as the analysis length
        assertEquals(YouTubePlayerState.UNSTARTED, YouTubePlayerState.of(42))
    }

    @Test fun videoTimeIsTheClockExtrapolatedBetweenReports() {
        val engine = engine()
        val bridge = FakeBridge()
        engine.attach(bridge)
        engine.play()
        bridge.state(YouTubePlayerState.PLAYING)
        bridge.time(2.0)
        advance(0.1)
        assertEquals(2.1, engine.position(), 1e-9)
        bridge.listener!!.onRateChange(0.5f)
        advance(0.2)
        assertEquals(2.2, engine.position(), 1e-9)
        bridge.time(2.25)
        assertEquals(2.25, engine.position(), 1e-9)
        // A stalled page never runs the clock away.
        advance(30.0)
        assertEquals(2.25 + YouTubePlaybackEngine.MAX_EXTRAPOLATION_SEC * 0.5, engine.position(), 1e-9)
        // Paused: the clock holds.
        engine.pause()
        val held = engine.position()
        advance(1.0)
        assertEquals(held, engine.position(), 0.0)
    }

    @Test fun idleReportsAndStaleReportsAfterASeekAreIgnored() {
        val engine = engine()
        val bridge = FakeBridge()
        engine.attach(bridge)
        engine.seekTo(6.0)
        bridge.time(0.0) // cued video says 0
        assertEquals(6.0, engine.position(), 0.0)

        bridge.state(YouTubePlayerState.PLAYING)
        engine.seekTo(8.0)
        bridge.time(6.2) // sent before the seek landed
        assertEquals(8.0, engine.position(), 1e-9)
        bridge.time(8.05)
        assertEquals(8.05, engine.position(), 1e-9)
        advance(2.0)
        bridge.time(3.0) // later reports are trusted again
        assertEquals(3.0, engine.position(), 1e-9)
    }

    @Test fun seekingMovesTheVideo() {
        val engine = engine()
        val bridge = FakeBridge()
        engine.attach(bridge)
        bridge.commands.clear()
        engine.seekTo(-1.0)
        engine.seekTo(7.25)
        assertEquals(listOf("seek 0.0", "seek 7.25"), bridge.commands)
    }

    @Test fun aNewBridgeTakesOverAndAStaleDetachIsIgnored() {
        val engine = engine()
        val first = FakeBridge()
        engine.attach(first)
        engine.setSpeed(0.75f)
        engine.play()
        first.state(YouTubePlayerState.PLAYING)
        first.time(5.0)

        val second = FakeBridge()
        engine.attach(second) // e.g. player → piano view
        assertEquals("pause", first.commands.last())
        assertNull(first.listener)
        assertEquals(listOf("rate 0.75", "seek 5.0", "play"), second.commands)
        assertTrue(engine.isPlaying)

        engine.detach(first) // the old screen leaving afterwards
        assertTrue(engine.attached)
        assertTrue(engine.isPlaying)

        engine.detach(second)
        assertFalse(engine.attached)
        assertFalse(engine.isPlaying)
        assertEquals(5.0, engine.position(), 1e-9)
    }

    @Test fun mutedFollowerMutesEveryBridge() {
        val engine = engine(muted = true)
        val bridge = FakeBridge()
        engine.attach(bridge)
        assertEquals(listOf("mute"), bridge.commands)
    }

    // --- VideoSync (local audio is the clock) ---

    private fun followerSetUp(): Triple<YouTubePlaybackEngine, FakeBridge, VideoSync> {
        val video = engine(muted = true)
        val bridge = FakeBridge()
        video.attach(bridge)
        bridge.state(YouTubePlayerState.CUED)
        bridge.commands.clear()
        return Triple(video, bridge, VideoSync(video) { now })
    }

    @Test fun syncMirrorsPlayPauseAndSpeed() {
        val (video, bridge, sync) = followerSetUp()
        sync.sync(0.0, audioPlaying = true, speed = 0.75f)
        assertTrue(video.isPlaying)
        assertEquals(listOf("rate 0.75", "seek 0.0", "play"), bridge.commands)
        bridge.commands.clear()
        bridge.state(YouTubePlayerState.PLAYING)
        sync.sync(0.0, audioPlaying = false, speed = 0.75f)
        assertFalse(video.isPlaying)
        assertEquals(listOf("pause"), bridge.commands)
    }

    @Test fun driftWithin150msIsLeftAlone() {
        val (_, bridge, sync) = followerSetUp()
        bridge.state(YouTubePlayerState.PLAYING)
        bridge.time(10.0)
        sync.sync(10.149, audioPlaying = false, speed = 1f)
        sync.sync(9.851, audioPlaying = false, speed = 1f)
        assertTrue(bridge.commands.none { it.startsWith("seek") })
    }

    @Test fun driftOver150msReseeksToTheAudioThenCoolsDown() {
        val (video, bridge, sync) = followerSetUp()
        bridge.state(YouTubePlayerState.PAUSED)
        bridge.time(10.0)
        sync.sync(10.2, audioPlaying = false, speed = 1f)
        assertEquals(listOf("seek 10.2"), bridge.commands)
        assertEquals(10.2, video.position(), 1e-9)

        // Still far off right after (the video hasn't caught up): no seek storm.
        advance(1.5)
        bridge.time(10.0)
        advance(-1.0) // report arrives within the cooldown
        sync.sync(12.0, audioPlaying = false, speed = 1f)
        assertEquals(1, bridge.commands.count { it.startsWith("seek") })

        advance(1.1)
        sync.sync(12.0, audioPlaying = false, speed = 1f)
        assertEquals(listOf("seek 10.2", "seek 12.0"), bridge.commands)
    }

    @Test fun noDriftJudgedWhileBufferingOrDetached() {
        val (video, bridge, sync) = followerSetUp()
        bridge.state(YouTubePlayerState.PLAYING)
        bridge.state(YouTubePlayerState.BUFFERING)
        sync.sync(30.0, audioPlaying = true, speed = 1f)
        assertTrue(bridge.commands.none { it.startsWith("seek") })

        video.detach(bridge)
        bridge.commands.clear()
        sync.sync(50.0, audioPlaying = true, speed = 1f)
        assertTrue(bridge.commands.isEmpty())
    }

    // --- Player integration: the video is the clock ---

    private val youTubeAnalysis = SyntheticAnalysis.make().let {
        it.copy(meta = it.meta.copy(source = Source.YOUTUBE, sourceRef = "dQw4w9WgXcQ"))
    }

    @Test fun playerUsesTheVideoClockAndScrubbingSeeksTheVideo() {
        val engine = engine()
        val vm = PlayerViewModel(
            loadTrack = { PlayerTrack(youTubeAnalysis, "Song", null, favorite = false, audio = null, videoId = "dQw4w9WgXcQ") },
            voicings = Contracts.voicings(),
            createEngine = { engine },
        )
        assertTrue(vm.state.value.videoIsClock)
        assertEquals(engine, vm.videoEngine)

        val bridge = FakeBridge()
        engine.attach(bridge)
        vm.seekTo(4.0)
        assertEquals("seek 4.0", bridge.commands.last())

        vm.togglePlay()
        bridge.state(YouTubePlayerState.PLAYING)
        bridge.time(4.0)
        advance(0.5)
        vm.tick()
        assertEquals(4.5, vm.position.value, 1e-9)
        assertTrue(vm.state.value.isPlaying)

        // Pausing from the embed reaches the player.
        engine.onPlayStateChange = vm::tick
        bridge.state(YouTubePlayerState.PAUSED)
        assertFalse(vm.state.value.isPlaying)
    }
}
