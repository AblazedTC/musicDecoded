package com.musicdecoded.mobile.player

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.BuildConfig
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.core.Accidentals
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.ChordVoicings
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.core.barRange
import com.musicdecoded.mobile.core.nextChordStart
import com.musicdecoded.mobile.core.previousChordStart
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.settings.PlayerPreferences
import com.musicdecoded.mobile.ui.library.SongActions
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.time.Instant
import kotlin.math.roundToInt

/** Everything the player needs about one track. [audio] is `null` when it isn't on this device. */
data class PlayerTrack(
    val analysis: Analysis,
    val title: String,
    val artist: String?,
    val favorite: Boolean,
    val audio: File?,
    /** YouTube videoId for youtube tracks (the embed in the Video panel). */
    val videoId: String? = null,
)

enum class ChordsMode { BASIC, FULL }

/** Why the player screen should close (header ⋯ menu). */
enum class PlayerExit { DELETED, REANALYZING }
enum class Visualizer { PIANO, VIDEO }

data class PlayerUiState(
    val loading: Boolean = true,
    val track: PlayerTrack? = null,
    val favorite: Boolean = false,
    val isPlaying: Boolean = false,
    val speed: Float = 1f,
    /** Loop region (current bar or a dragged, beat-snapped range); `null` = off. */
    val loop: OpenEndRange<Double>? = null,
    val chordsMode: ChordsMode = ChordsMode.FULL,
    /** Semitones, −6…+6. Display only in v1 (not audio pitch). */
    val transpose: Int = 0,
    val showNoteNames: Boolean = true,
    val visualizer: Visualizer = Visualizer.PIANO,
    val voicings: ChordVoicings? = null,
    /** Settings → Accidentals. */
    val accidentals: Accidentals = Accidentals.AUTO,
    /** One-shot: leave the player (after Delete / Re-analyze from the ⋯ menu). */
    val exit: PlayerExit? = null,
    /** One-shot user-facing error from a menu action. */
    val message: String? = null,
) {
    val analysis: Analysis? get() = track?.analysis

    /** Video is only for youtube sources (DESIGN.md §3.7 item 6). */
    val videoAvailable: Boolean get() = analysis?.meta?.source == Source.YOUTUBE && track?.videoId != null

    /** No local audio: the video's `currentTime` is the clock (DESIGN.md §4), so it stays on screen. */
    val videoIsClock: Boolean get() = videoAvailable && track?.audio == null

    val chords: ChordDisplay get() = ChordDisplay(analysis?.key, transpose, chordsMode == ChordsMode.BASIC, voicings, accidentals)
}

/**
 * State for the Player and Piano learning view (shared; the piano view reuses the player's
 * instance). The [PlaybackEngine] position is the clock: the UI calls [tick] every frame while
 * playing (~60 Hz), which publishes [position] and enforces the loop.
 */
class PlayerViewModel(
    private val loadTrack: suspend () -> PlayerTrack?,
    private val voicings: ChordVoicings?,
    private val createEngine: (PlayerTrack) -> PlaybackEngine,
    private val saveFavorite: suspend (Boolean) -> Unit = {},
    /** `false` when the engine belongs to the app-level [NowPlaying] (playback outlives this screen). */
    private val ownsEngine: Boolean = true,
    /** Settings defaults (chords, accidentals, visualizer); default speed is applied by [createEngine]. */
    private val preferences: suspend () -> PlayerPreferences = { PlayerPreferences() },
    /** ⋯ Delete: removes the library item (and clears the mini player). */
    private val deleteTrack: suspend () -> Unit = {},
    /** ⋯ Re-analyze: requests a new analysis (TranscriptionRepository.retry). */
    private val reanalyzeTrack: suspend () -> Unit = {},
    /** Playback went from stopped to playing: stamps `LibraryItem.lastPlayedAt` (+ updatedAt). */
    private val markPlayed: suspend () -> Unit = {},
) : ViewModel() {
    private val _state = MutableStateFlow(PlayerUiState(voicings = voicings))
    val state: StateFlow<PlayerUiState> = _state.asStateFlow()

    private val _position = MutableStateFlow(0.0)

    /** Playhead in seconds. */
    val position: StateFlow<Double> = _position.asStateFlow()

    private var engine: PlaybackEngine? = null

    /** The embedded video when it's the clock (youtube track, no local audio); `null` otherwise. */
    val videoEngine: YouTubePlaybackEngine? get() = engine as? YouTubePlaybackEngine

    init {
        viewModelScope.launch {
            val prefs = runCatching { preferences() }.getOrDefault(PlayerPreferences())
            val track = runCatching { loadTrack() }.getOrNull()
            if (track != null) engine = createEngine(track)
            _state.update {
                val loaded = it.copy(loading = false, track = track, favorite = track?.favorite ?: false, speed = engine?.speed ?: it.speed)
                loaded.copy(
                    chordsMode = prefs.chordsMode,
                    accidentals = prefs.accidentals,
                    visualizer = if (prefs.visualizer == Visualizer.VIDEO && loaded.videoAvailable) Visualizer.VIDEO else Visualizer.PIANO,
                )
            }
            tick() // picks up a track that's still playing from an earlier visit (NowPlaying)
        }
    }

    private val duration: Double get() = _state.value.analysis?.meta?.durationSec ?: 0.0

    /** Reads the engine clock: publishes the position, wraps the loop, stops at the end. */
    fun tick() {
        val engine = engine ?: return
        var t = engine.position()
        val loop = _state.value.loop
        if (loop != null && engine.isPlaying && (t >= loop.endExclusive || t < loop.start - LOOP_ENTRY_TOLERANCE)) {
            engine.seekTo(loop.start)
            t = loop.start
        } else if (loop == null && t >= duration && engine.isPlaying) {
            engine.pause()
            t = duration
        }
        _position.value = t
        publishPlaying(engine.isPlaying)
    }

    /** Publishes [playing]; a stopped → playing change counts as a play (lastPlayedAt). */
    private fun publishPlaying(playing: Boolean) {
        if (playing == _state.value.isPlaying) return
        _state.update { it.copy(isPlaying = playing) }
        if (playing) viewModelScope.launch { runCatching { markPlayed() } }
    }

    fun togglePlay() {
        val engine = engine ?: return
        if (engine.isPlaying) {
            engine.pause()
        } else {
            if (_position.value >= duration - END_TOLERANCE) seekTo(_state.value.loop?.start ?: 0.0)
            engine.play()
        }
        publishPlaying(engine.isPlaying)
        tick()
    }

    fun pause() {
        val engine = engine ?: return
        if (engine.isPlaying) togglePlay()
    }

    fun seekTo(seconds: Double) {
        val engine = engine ?: return
        val t = seconds.coerceIn(0.0, duration)
        engine.seekTo(t)
        _position.value = t
    }

    fun previousChord() {
        val analysis = _state.value.analysis ?: return
        previousChordStart(_position.value, analysis)?.let(::seekTo)
    }

    fun nextChord() {
        val analysis = _state.value.analysis ?: return
        nextChordStart(_position.value, analysis)?.let(::seekTo)
    }

    /** Snaps to 0.05 steps within 0.5–1.25×. */
    fun setSpeed(speed: Float) {
        val snapped = ((speed * 20).roundToInt() / 20f).coerceIn(MIN_SPEED, MAX_SPEED)
        engine?.setSpeed(snapped)
        _state.update { it.copy(speed = snapped) }
    }

    /** Tap on the speed button: 1.00× ⇄ 0.75×. */
    fun toggleSpeed() = setSpeed(if (_state.value.speed == 1f) PRACTICE_SPEED else 1f)

    /** Loop off → loop the current bar; loop on → clear it. */
    fun toggleLoop() {
        if (_state.value.loop != null) {
            clearLoop()
        } else {
            val analysis = _state.value.analysis ?: return
            barRange(_position.value, analysis)?.let(::setLoop)
        }
    }

    /** A custom range (from dragging on the chord track); the playhead jumps in if it's outside. */
    fun setLoop(range: OpenEndRange<Double>) {
        _state.update { it.copy(loop = range) }
        if (_position.value !in range) seekTo(range.start)
    }

    fun clearLoop() = _state.update { it.copy(loop = null) }

    fun toggleChordsMode() = _state.update {
        it.copy(chordsMode = if (it.chordsMode == ChordsMode.BASIC) ChordsMode.FULL else ChordsMode.BASIC)
    }

    fun setTranspose(semitones: Int) = _state.update { it.copy(transpose = semitones.coerceIn(-MAX_TRANSPOSE, MAX_TRANSPOSE)) }

    fun toggleNoteNames() = _state.update { it.copy(showNoteNames = !it.showNoteNames) }

    fun selectVisualizer(visualizer: Visualizer) {
        if (visualizer == Visualizer.VIDEO && !_state.value.videoAvailable) return
        _state.update { it.copy(visualizer = visualizer) }
    }

    fun toggleFavorite() {
        val favorite = !_state.value.favorite
        _state.update { it.copy(favorite = favorite) }
        viewModelScope.launch { runCatching { saveFavorite(favorite) } }
    }

    fun delete() = menuAction(PlayerExit.DELETED) {
        pause()
        deleteTrack()
    }

    fun reanalyze() = menuAction(PlayerExit.REANALYZING) { reanalyzeTrack() }

    fun exitHandled() = _state.update { it.copy(exit = null) }

    fun messageShown() = _state.update { it.copy(message = null) }

    private fun menuAction(exit: PlayerExit, block: suspend () -> Unit) {
        viewModelScope.launch {
            try {
                block()
                _state.update { it.copy(exit = exit) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(message = TranscriptionRepository.userMessage(e)) }
            }
        }
    }

    override fun onCleared() {
        if (ownsEngine) engine?.release()
        engine = null
    }

    companion object {
        const val MIN_SPEED = 0.5f
        const val MAX_SPEED = 1.25f
        const val PRACTICE_SPEED = 0.75f
        const val MAX_TRANSPOSE = 6

        /** Long-press speed choices: 0.50…1.25 in 0.05 steps. */
        val SPEED_STEPS: List<Float> = (10..25).map { it / 20f }

        /** A position this close to the end counts as "ended", so play restarts from the top. */
        private const val END_TOLERANCE = 0.05

        /** Seeking slightly before a loop (e.g. snapping) doesn't count as leaving it. */
        private const val LOOP_ENTRY_TOLERANCE = 0.25

        fun factory(app: MusicDecodedApplication, trackId: String): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val dao = app.libraryDatabase.libraryDao()
                var defaultSpeed = 1f // read before createEngine runs (preferences load first)
                PlayerViewModel(
                    loadTrack = {
                        withContext(Dispatchers.IO) {
                            val item = dao.get(trackId)
                            val analysis = app.analysisStore.load(trackId)
                                ?: if (BuildConfig.DEBUG) DebugFixtures.load(app, trackId) else null
                            analysis?.let {
                                PlayerTrack(
                                    analysis = it,
                                    title = item?.title ?: trackId,
                                    artist = item?.artist,
                                    favorite = item?.favorite ?: false,
                                    audio = localAudio(app, trackId, item?.localAudioPath),
                                    videoId = (item?.sourceRef ?: it.meta.sourceRef).takeIf { _ -> it.meta.source == Source.YOUTUBE },
                                )
                            }
                        }
                    },
                    voicings = runCatching { loadChordVoicings(app) }.getOrNull(),
                    createEngine = { track ->
                        app.nowPlaying.engineFor(trackId, track) {
                            val duration = it.analysis.meta.durationSec
                            when {
                                it.audio != null -> ExoPlaybackEngine(app, it.audio, duration)
                                it.videoId != null -> YouTubePlaybackEngine(duration)
                                else -> SilentPlaybackEngine(duration)
                            }.apply { setSpeed(defaultSpeed) } // a new engine starts at Settings → Default speed
                        }
                    },
                    saveFavorite = { favorite ->
                        if (dao.get(trackId) != null) dao.setFavorite(trackId, favorite, Instant.now())
                    },
                    ownsEngine = false,
                    preferences = { app.settingsStore.current().also { defaultSpeed = it.defaultSpeed } },
                    deleteTrack = { SongActions.from(app).delete(trackId) },
                    reanalyzeTrack = { app.transcriptionRepository.retry(trackId) },
                    markPlayed = { dao.setLastPlayed(trackId, Instant.now()) },
                )
            }
        }

        /** `LibraryItem.localAudioPath` (relative to filesDir), else `files/audio/{trackId}.*` (PLAN.md §5.1). */
        private fun localAudio(app: MusicDecodedApplication, trackId: String, path: String?): File? =
            path?.let { File(app.filesDir, it) }?.takeIf { it.isFile }
                ?: File(app.filesDir, "audio").listFiles { file -> file.nameWithoutExtension == trackId && file.isFile }?.firstOrNull()
    }
}
