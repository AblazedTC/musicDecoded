package com.musicdecoded.mobile.player

import com.musicdecoded.mobile.core.Analysis
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * The app-level "now playing" session: owns the [PlaybackEngine] of the last track opened in the
 * player, so playback keeps going when the player is closed and the mini player (DESIGN.md §2) can
 * show and control it from Home/Library. Call from the main thread.
 */
class NowPlaying {
    data class State(
        val trackId: String,
        val title: String,
        val artist: String?,
        val key: Analysis.Key?,
        val durationSec: Double,
        val isPlaying: Boolean,
    )

    private var engine: PlaybackEngine? = null
    private val _state = MutableStateFlow<State?>(null)
    private val _position = MutableStateFlow(0.0)

    /** `null` when nothing is loaded (no mini player). */
    val state: StateFlow<State?> = _state.asStateFlow()

    /** Seconds; updated by [tick]. */
    val position: StateFlow<Double> = _position.asStateFlow()

    /**
     * The engine for [trackId]: the running one if that track is already loaded, otherwise the
     * previous one is released and [create] makes a new one.
     */
    fun engineFor(trackId: String, track: PlayerTrack, create: (PlayerTrack) -> PlaybackEngine): PlaybackEngine {
        engine?.let { current -> if (_state.value?.trackId == trackId) return current }
        engine?.release()
        val created = create(track)
        engine = created
        _position.value = created.position()
        _state.value = State(trackId, track.title, track.artist, track.analysis.key, track.analysis.meta.durationSec, created.isPlaying)
        return created
    }

    /** Reads the engine clock; stops at the end of the track (loops only run inside the player). */
    fun tick() {
        val engine = engine ?: return
        val current = _state.value ?: return
        var t = engine.position()
        if (engine.isPlaying && t >= current.durationSec) {
            engine.pause()
            t = current.durationSec
        }
        _position.value = t
        if (engine.isPlaying != current.isPlaying) _state.value = current.copy(isPlaying = engine.isPlaying)
    }

    fun togglePlay() {
        val engine = engine ?: return
        val current = _state.value ?: return
        if (engine.isPlaying) {
            engine.pause()
        } else {
            if (engine.position() >= current.durationSec - END_TOLERANCE) engine.seekTo(0.0)
            engine.play()
        }
        tick()
    }

    fun pause() {
        if (engine?.isPlaying == true) togglePlay()
    }

    /** Stops and unloads [trackId] if it's the loaded track (e.g. it was just deleted). */
    fun dismiss(trackId: String) {
        if (_state.value?.trackId == trackId) dismiss()
    }

    /** The mini player's ×: stops and unloads the track. */
    fun dismiss() {
        engine?.release()
        engine = null
        _state.value = null
        _position.value = 0.0
    }

    private companion object {
        const val END_TOLERANCE = 0.05
    }
}
