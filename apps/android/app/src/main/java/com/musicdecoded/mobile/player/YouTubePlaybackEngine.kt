package com.musicdecoded.mobile.player

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * The embedded YouTube player as a [PlaybackEngine] (T-305a). For a youtube track with no local
 * audio the video's `currentTime` is the clock (DESIGN.md §4); with [muted] it's the follower
 * video kept in sync with local audio by [VideoSync].
 *
 * The video lives in a WebView on screen, so the engine only plays while a [YouTubePlayerBridge] is
 * [attach]ed; with none it keeps its last position and stays paused (no background video audio).
 * Attaching a second bridge (e.g. the player → piano view transition) hands playback over to it.
 *
 * Between the page's `currentTime` reports (a few per second) the position is extrapolated with
 * the video's actual rate, so the playhead stays smooth at 60 Hz.
 */
class YouTubePlaybackEngine(
    private val durationSec: Double,
    private val muted: Boolean = false,
    private val nanoTime: () -> Long = System::nanoTime,
) : PlaybackEngine, YouTubePlayerBridge.Listener {
    private var bridge: YouTubePlayerBridge? = null

    /** The last known position and when it was true. */
    private var anchor = 0.0
    private var anchorNanos = 0L

    /** The rate YouTube reports it's actually playing at (it may round the requested speed). */
    private var rate = 1f

    /** The last seek we asked for: stale reports from before it are ignored for a moment. */
    private var seekTarget: Double? = null
    private var seekNanos = 0L

    /** The embedded player's own state. */
    var videoState: YouTubePlayerState = YouTubePlayerState.UNSTARTED
        private set

    override var isPlaying: Boolean = false
        private set

    override var speed: Float = 1f
        private set

    /** Called when the video starts or stops on its own (e.g. a tap on the embed's controls). */
    var onPlayStateChange: (() -> Unit)? = null

    val attached: Boolean get() = bridge != null
    val buffering: Boolean get() = videoState == YouTubePlayerState.BUFFERING

    fun attach(newBridge: YouTubePlayerBridge) {
        if (bridge === newBridge) return
        bridge?.let { old ->
            old.listener = null
            old.pause()
        }
        reanchor()
        bridge = newBridge
        videoState = YouTubePlayerState.UNSTARTED
        rate = 1f
        newBridge.listener = this
        if (muted) newBridge.mute()
        if (speed != 1f) newBridge.setPlaybackRate(speed)
        if (isPlaying) start(newBridge)
    }

    /** Ignored unless [old] is the attached bridge (a newer one may already have taken over). */
    fun detach(old: YouTubePlayerBridge) {
        if (bridge !== old) return
        reanchor()
        old.listener = null
        bridge = null
        videoState = YouTubePlayerState.UNSTARTED
        setPlaying(false, notify = true)
    }

    override fun position(): Double = when {
        bridge == null -> anchor
        videoState == YouTubePlayerState.ENDED -> max(anchor, durationSec)
        // Only while we want it playing: after pause() the clock holds even before the page confirms.
        isPlaying && videoState == YouTubePlayerState.PLAYING -> {
            val elapsed = min((nanoTime() - anchorNanos) / 1e9, MAX_EXTRAPOLATION_SEC)
            anchor + elapsed * rate
        }
        else -> anchor
    }

    override fun play() {
        val bridge = bridge ?: return // no video on screen: nothing can play
        setPlaying(true, notify = false)
        start(bridge)
    }

    override fun pause() {
        reanchor()
        setPlaying(false, notify = false)
        bridge?.pause()
    }

    override fun seekTo(seconds: Double) {
        val t = max(seconds, 0.0)
        anchor = t
        anchorNanos = nanoTime()
        seekTarget = t
        seekNanos = anchorNanos
        bridge?.seekTo(t)
    }

    override fun setSpeed(speed: Float) {
        reanchor()
        this.speed = speed
        bridge?.setPlaybackRate(speed)
    }

    override fun release() {
        bridge?.let {
            it.listener = null
            it.pause()
        }
        bridge = null
        isPlaying = false
    }

    // --- Events from the embed (main thread) ---

    override fun onStateChange(state: YouTubePlayerState) {
        reanchor()
        videoState = state
        when (state) {
            YouTubePlayerState.PLAYING -> setPlaying(true, notify = true)
            YouTubePlayerState.PAUSED, YouTubePlayerState.ENDED -> setPlaying(false, notify = true)
            // Buffering keeps the intent; unstarted/cued are before playback (or after a cue).
            YouTubePlayerState.BUFFERING, YouTubePlayerState.UNSTARTED, YouTubePlayerState.CUED -> Unit
        }
    }

    override fun onTime(seconds: Double) {
        // A cued/unstarted video reports 0 (or its start), which says nothing about our position.
        if (videoState.idle) return
        val target = seekTarget
        if (target != null) {
            val sinceSeek = (nanoTime() - seekNanos) / 1e9
            if (sinceSeek < SEEK_SETTLE_SEC && abs(seconds - target) > SEEK_TOLERANCE_SEC) return
            seekTarget = null
        }
        anchor = seconds
        anchorNanos = nanoTime()
    }

    override fun onRateChange(rate: Float) {
        reanchor()
        this.rate = rate
    }

    /** An idle (unstarted/cued) video is cued at our position first so it starts there. */
    private fun start(bridge: YouTubePlayerBridge) {
        if (videoState.idle) bridge.seekTo(anchor)
        bridge.play()
    }

    private fun reanchor() {
        anchor = position()
        anchorNanos = nanoTime()
    }

    private fun setPlaying(playing: Boolean, notify: Boolean) {
        if (isPlaying == playing) return
        isPlaying = playing
        if (notify) onPlayStateChange?.invoke()
    }

    companion object {
        /** A stalled page never moves the clock more than this past its last report. */
        const val MAX_EXTRAPOLATION_SEC = 1.0

        /** Reports this far from a fresh seek target are from before the seek. */
        const val SEEK_TOLERANCE_SEC = 1.0
        const val SEEK_SETTLE_SEC = 1.0
    }
}

/**
 * Keeps a muted [video] within [MAX_DRIFT_SEC] of the local audio clock (T-305a, flag-on case):
 * mirrors play/pause and speed, and re-seeks the video when the drift is larger. After a seek it
 * waits [SEEK_COOLDOWN_SEC] so a video that's still catching up isn't re-seeked every frame, and
 * it never judges drift while the video is buffering.
 */
class VideoSync(
    private val video: YouTubePlaybackEngine,
    private val nanoTime: () -> Long = System::nanoTime,
) {
    private var lastSeekNanos: Long? = null

    /** Call with the audio clock on every player tick (and after seeks while paused). */
    fun sync(audioPosition: Double, audioPlaying: Boolean, speed: Float) {
        if (!video.attached) return
        if (video.speed != speed) video.setSpeed(speed)
        if (audioPlaying && !video.isPlaying) video.play()
        if (!audioPlaying && video.isPlaying) video.pause()
        if (video.buffering) return
        val drift = abs(video.position() - audioPosition)
        if (drift <= MAX_DRIFT_SEC) return
        val now = nanoTime()
        val last = lastSeekNanos
        if (last != null && (now - last) / 1e9 < SEEK_COOLDOWN_SEC) return
        video.seekTo(audioPosition)
        lastSeekNanos = now
    }

    companion object {
        const val MAX_DRIFT_SEC = 0.150
        const val SEEK_COOLDOWN_SEC = 1.0
    }
}
