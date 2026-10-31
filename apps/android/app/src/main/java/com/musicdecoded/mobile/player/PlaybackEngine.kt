package com.musicdecoded.mobile.player

import android.content.Context
import android.net.Uri
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import java.io.File

/** What the player's clock reads from. All times are seconds. Call from the main thread. */
interface PlaybackEngine {
    /** Current playback position. */
    fun position(): Double

    /** `true` while playing (or about to, e.g. buffering). The engine may stop on its own, e.g. on audio focus loss. */
    val isPlaying: Boolean

    fun play()
    fun pause()
    fun seekTo(seconds: Double)

    /** The current speed (the player reads it back when it reopens a track that's still loaded). */
    val speed: Float get() = 1f

    /** 0.5–1.25×, pitch kept. */
    fun setSpeed(speed: Float)
    fun release()
}

/**
 * Plays the local `audio/{trackId}` file with Media3 ExoPlayer. Once the media has ended the
 * position reads as [durationSec] (the analysis length), so the clock's end/loop logic always fires.
 */
class ExoPlaybackEngine(context: Context, audio: File, private val durationSec: Double) : PlaybackEngine {
    private val player = ExoPlayer.Builder(context)
        .setAudioAttributes(
            AudioAttributes.Builder().setUsage(C.USAGE_MEDIA).setContentType(C.AUDIO_CONTENT_TYPE_MUSIC).build(),
            /* handleAudioFocus = */ true,
        )
        .setHandleAudioBecomingNoisy(true)
        .build()
        .apply {
            setMediaItem(MediaItem.fromUri(Uri.fromFile(audio)))
            prepare()
        }

    private val smoother = PositionSmoother()

    override fun position(): Double {
        val raw = player.currentPosition / 1000.0
        return when {
            player.playbackState == Player.STATE_ENDED -> maxOf(durationSec, raw)
            // Between ExoPlayer's coarse position updates (~4 Hz on some devices) the clock is
            // extrapolated, so the 60 Hz timeline moves every frame instead of in jumps.
            player.isPlaying -> smoother.position(raw, System.nanoTime(), player.playbackParameters.speed)
            else -> raw.also { smoother.reset() }
        }
    }
    override val isPlaying: Boolean get() = player.playWhenReady
    override fun play() = player.play().also { smoother.reset() }
    override fun pause() = player.pause().also { smoother.reset() }
    override fun seekTo(seconds: Double) = player.seekTo((seconds * 1000).toLong()).also { smoother.reset() }

    override val speed: Float get() = player.playbackParameters.speed

    // Time-stretch without changing pitch (DESIGN.md §4).
    override fun setSpeed(speed: Float) {
        player.playbackParameters = PlaybackParameters(speed, 1f)
        smoother.reset()
    }

    override fun release() = player.release()
}

/**
 * Turns a position that only updates in steps (e.g. every ~250 ms) into a smooth clock: between
 * steps it extrapolates in real time × speed (at most [maxLead] past the last step, in case playback
 * stalls), and it never steps back by less than [backTolerance], because the next step often lands
 * just behind the estimate. Bigger jumps (seeks) pass through. Call [reset] on play/pause/seek/speed.
 */
class PositionSmoother(private val maxLead: Double = 0.5, private val backTolerance: Double = 0.25) {
    private var stepValue = Double.NaN
    private var stepNanos = 0L
    private var last = Double.NaN

    fun reset() {
        stepValue = Double.NaN
        last = Double.NaN
    }

    fun position(raw: Double, nowNanos: Long, speed: Float): Double {
        if (raw != stepValue) {
            stepValue = raw
            stepNanos = nowNanos
        }
        var t = raw + ((nowNanos - stepNanos) / 1e9 * speed).coerceIn(0.0, maxLead)
        if (!last.isNaN() && t < last && last - t < backTolerance) t = last
        last = t
        return t
    }
}

/**
 * A clock with no audio, for file/recording tracks whose audio isn't on this device and for the
 * debug fixture (youtube tracks use [YouTubePlaybackEngine]). Advances in real time × speed; [nanoTime] is
 * injectable so tests can drive it.
 */
class SilentPlaybackEngine(
    private val durationSec: Double,
    private val nanoTime: () -> Long = System::nanoTime,
) : PlaybackEngine {
    private var anchorPosition = 0.0
    private var anchorNanos = 0L
    override var speed = 1f
        private set

    override var isPlaying: Boolean = false
        private set

    override fun position(): Double {
        if (!isPlaying) return anchorPosition
        val elapsed = (nanoTime() - anchorNanos) / 1e9 * speed
        return (anchorPosition + elapsed).coerceAtMost(durationSec)
    }

    override fun play() {
        anchorNanos = nanoTime()
        isPlaying = true
    }

    override fun pause() {
        anchorPosition = position()
        isPlaying = false
    }

    override fun seekTo(seconds: Double) {
        anchorPosition = seconds.coerceIn(0.0, durationSec)
        anchorNanos = nanoTime()
    }

    override fun setSpeed(speed: Float) {
        anchorPosition = position()
        anchorNanos = nanoTime()
        this.speed = speed
    }

    override fun release() = Unit
}
