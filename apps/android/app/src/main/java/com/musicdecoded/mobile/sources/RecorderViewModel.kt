package com.musicdecoded.mobile.sources

import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LocalAudio
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.library.TranscriptionRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

enum class RecorderPhase {
    /** Microphone ready, nothing recorded yet. */
    IDLE,
    RECORDING,
    PAUSED,
    /** Recording finished and kept; waiting for a title (or the title prompt was dismissed). */
    STOPPED,
    /** Uploading + requesting the job. */
    STARTING,
}

data class RecorderUiState(
    val phase: RecorderPhase = RecorderPhase.IDLE,
    val elapsedMs: Long = 0,
    val levelDb: Float = InputLevel.MIN_DB,
    /** Recent levels, 0–1, oldest first. */
    val waveform: List<Float> = emptyList(),
    val askTitle: Boolean = false,
    val defaultTitle: String = "",
    val error: String? = null,
    /** Set once the job has started. */
    val startedTrackId: String? = null,
) {
    val hasRecording: Boolean get() = phase != RecorderPhase.IDLE
}

/** Record source (T-206a): MediaRecorder → title → [TranscriptionRepository.start]. Max 10 minutes. */
class RecorderViewModel(
    private val recorder: AudioRecorder,
    private val repository: TranscriptionRepository,
    private val appScope: CoroutineScope,
) : ViewModel() {
    private val _state = MutableStateFlow(RecorderUiState())
    val state: StateFlow<RecorderUiState> = _state.asStateFlow()

    private var file: File? = null
    private var ticker: Job? = null
    /** Time recorded before the current segment (pauses split segments). */
    private var recordedBeforeMs = 0L
    private var segmentStartedAt = 0L

    /** The big button: start, pause or resume. */
    fun toggleRecord() {
        when (_state.value.phase) {
            RecorderPhase.IDLE -> startRecording()
            RecorderPhase.RECORDING -> pause()
            RecorderPhase.PAUSED -> resume()
            RecorderPhase.STOPPED, RecorderPhase.STARTING -> Unit
        }
    }

    /** Pauses when the app leaves the foreground: background apps can't use the mic. */
    fun onBackgrounded() {
        if (_state.value.phase == RecorderPhase.RECORDING) pause()
    }

    fun discard() {
        if (_state.value.phase == RecorderPhase.STARTING) return
        stopTicker()
        recorder.release()
        file?.delete()
        file = null
        recordedBeforeMs = 0
        _state.value = RecorderUiState()
    }

    /** Finish ✓ / Save: stop the recording and ask for a title. */
    fun finish() {
        when (_state.value.phase) {
            RecorderPhase.RECORDING, RecorderPhase.PAUSED -> {
                if (_state.value.phase == RecorderPhase.RECORDING) recordedBeforeMs += SystemClock.elapsedRealtime() - segmentStartedAt
                stopTicker()
                if (!recorder.stop()) {
                    discard()
                    _state.update { it.copy(error = "That recording was too short. Try again.") }
                    return
                }
                _state.update {
                    it.copy(phase = RecorderPhase.STOPPED, elapsedMs = recordedBeforeMs, levelDb = InputLevel.MIN_DB, askTitle = true, defaultTitle = defaultTitle())
                }
            }
            RecorderPhase.STOPPED -> _state.update { it.copy(askTitle = true) }
            RecorderPhase.IDLE, RecorderPhase.STARTING -> Unit
        }
    }

    fun dismissTitle() = _state.update { it.copy(askTitle = false) }

    fun save(title: String) {
        val audio = file ?: return
        if (_state.value.phase != RecorderPhase.STOPPED) return
        val name = title.trim().ifEmpty { _state.value.defaultTitle }
        _state.update { it.copy(phase = RecorderPhase.STARTING, askTitle = false, error = null) }
        appScope.launch {
            try {
                val trackId = repository.start(LocalAudio(audio, name, Source.RECORDING, recordedBeforeMs / 1000.0))
                file = null
                _state.update { it.copy(startedTrackId = trackId) }
            } catch (e: TranscriptionException) {
                _state.update { it.copy(phase = RecorderPhase.STOPPED, error = e.message) }
            }
        }
    }

    override fun onCleared() {
        stopTicker()
        recorder.release()
        if (_state.value.phase != RecorderPhase.STARTING) file?.delete()
    }

    private fun startRecording() {
        val target = repository.newPendingFile("m4a")
        try {
            recorder.start(target) { viewModelScope.launch { finish() } }
        } catch (e: Exception) {
            Log.w("Recorder", "Couldn't start recording", e)
            recorder.release()
            target.delete()
            _state.update { it.copy(error = "Couldn't start the microphone. Close other apps using it and try again.") }
            return
        }
        file = target
        recordedBeforeMs = 0
        _state.value = RecorderUiState(phase = RecorderPhase.RECORDING)
        startTicker()
    }

    private fun pause() {
        recorder.pause()
        recordedBeforeMs += SystemClock.elapsedRealtime() - segmentStartedAt
        stopTicker()
        _state.update { it.copy(phase = RecorderPhase.PAUSED, elapsedMs = recordedBeforeMs, levelDb = InputLevel.MIN_DB) }
    }

    private fun resume() {
        recorder.resume()
        _state.update { it.copy(phase = RecorderPhase.RECORDING) }
        startTicker()
    }

    private fun startTicker() {
        segmentStartedAt = SystemClock.elapsedRealtime()
        ticker = viewModelScope.launch {
            while (isActive) {
                val elapsed = recordedBeforeMs + SystemClock.elapsedRealtime() - segmentStartedAt
                val db = InputLevel.dbfs(recorder.amplitude())
                _state.update {
                    it.copy(elapsedMs = elapsed, levelDb = db, waveform = (it.waveform + InputLevel.normalized(db)).takeLast(WAVEFORM_BARS))
                }
                if (elapsed >= AudioRecorder.MAX_DURATION_MS) {
                    finish()
                    break
                }
                delay(TICK_MS)
            }
        }
    }

    private fun stopTicker() {
        ticker?.cancel()
        ticker = null
    }

    private fun defaultTitle() = "Recording " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("MMM d, HH:mm"))

    companion object {
        const val WAVEFORM_BARS = 48
        private const val TICK_MS = 60L
    }
}
