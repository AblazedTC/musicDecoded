package com.musicdecoded.mobile.sources

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import java.io.File
import kotlin.math.log10
import kotlin.math.roundToInt

/** MediaRecorder set up per PLAN.md §3.2: AAC, 44.1 kHz, mono, in an MPEG-4 (`.m4a`) file. */
class AudioRecorder(private val context: Context) {
    private var recorder: MediaRecorder? = null

    /** Starts recording into [file]; [onMaxDuration] fires when [MAX_DURATION_MS] is reached. */
    fun start(file: File, onMaxDuration: () -> Unit) {
        release()
        val mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) MediaRecorder(context) else @Suppress("DEPRECATION") MediaRecorder()
        recorder = mediaRecorder
        mediaRecorder.apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setAudioSamplingRate(SAMPLE_RATE)
            setAudioChannels(1)
            setAudioEncodingBitRate(BIT_RATE)
            setMaxDuration(MAX_DURATION_MS.toInt())
            setOnInfoListener { _, what, _ ->
                if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED) onMaxDuration()
            }
            setOutputFile(file.absolutePath)
            prepare()
            start()
        }
    }

    fun pause() { recorder?.pause() }

    fun resume() { recorder?.resume() }

    /** Peak amplitude (0–32767) since the last call. */
    fun amplitude(): Int = recorder?.maxAmplitude ?: 0

    /** Stops and finalizes the file. Returns false if nothing usable was recorded. */
    fun stop(): Boolean {
        val mediaRecorder = recorder ?: return false
        return try {
            mediaRecorder.stop()
            true
        } catch (_: RuntimeException) {
            false // stop() right after start() throws: no audio was captured
        } finally {
            release()
        }
    }

    fun release() {
        recorder?.release()
        recorder = null
    }

    companion object {
        const val SAMPLE_RATE = 44_100
        const val BIT_RATE = 128_000
        const val MAX_DURATION_MS = 10 * 60 * 1000L
    }
}

/** Input-level math for the recorder's meter (DESIGN.md §3.6). */
object InputLevel {
    const val SEGMENTS = 16
    /** The meter's floor; quieter input reads as this. */
    const val MIN_DB = -60f
    private const val FULL_SCALE = 32_767f

    /** Peak amplitude → dBFS, clamped to [[MIN_DB], 0]. */
    fun dbfs(amplitude: Int): Float =
        if (amplitude <= 0) MIN_DB else (20 * log10(amplitude / FULL_SCALE)).coerceIn(MIN_DB, 0f)

    /** How many of the [SEGMENTS] meter segments are lit for [db]. */
    fun litSegments(db: Float): Int = ((db - MIN_DB) / -MIN_DB * SEGMENTS).roundToInt().coerceIn(0, SEGMENTS)

    /** 0–1 bar height for the waveform. */
    fun normalized(db: Float): Float = ((db - MIN_DB) / -MIN_DB).coerceIn(0f, 1f)

    /** `00:MM:SS`. */
    fun formatTimer(ms: Long): String {
        val total = ms / 1000
        return "%02d:%02d:%02d".format(total / 3600, (total / 60) % 60, total % 60)
    }
}
