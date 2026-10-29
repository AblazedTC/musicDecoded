package com.musicdecoded.mobile.sources.youtube

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.network.VideoSummary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class YouTubeDetailUiState(
    val video: VideoSummary,
    /** This video's row in the library, if it was decoded (or is decoding) before. */
    val existing: LibraryItem? = null,
    /** The hero thumbnail was tapped: the embed plays inline. */
    val playing: Boolean = false,
    /** `POST /jobs` in flight. */
    val starting: Boolean = false,
    val error: String? = null,
    /** One-shot: the job exists, open Processing for this trackId. */
    val startedTrackId: String? = null,
) {
    val alreadyDecoded: Boolean get() = existing?.status == LibraryStatus.READY
}

/** Video detail (DESIGN.md §3.4): "Decode this song" → `youtube` job → Processing. */
class YouTubeDetailViewModel(
    video: VideoSummary,
    existing: Flow<LibraryItem?>,
    private val startJob: suspend (VideoSummary) -> String,
    /** Outlives this screen, so leaving it doesn't cancel a request in flight. */
    private val appScope: CoroutineScope,
) : ViewModel() {
    private val _state = MutableStateFlow(YouTubeDetailUiState(video))
    val state: StateFlow<YouTubeDetailUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch { existing.collect { item -> _state.update { it.copy(existing = item) } } }
    }

    fun play() = _state.update { it.copy(playing = true) }

    fun decode() {
        if (_state.value.starting) return
        _state.update { it.copy(starting = true, error = null) }
        appScope.launch {
            try {
                val trackId = startJob(_state.value.video)
                _state.update { it.copy(starting = false, startedTrackId = trackId) }
            } catch (e: TranscriptionException) {
                _state.update { it.copy(starting = false, error = e.message) }
            }
        }
    }

    fun navigated() = _state.update { it.copy(startedTrackId = null) }

    companion object {
        fun factory(app: MusicDecodedApplication, video: VideoSummary): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                YouTubeDetailViewModel(
                    video,
                    app.libraryDatabase.libraryDao().observeBySourceRef(Source.YOUTUBE.name, video.videoId),
                    startJob = {
                        app.transcriptionRepository.startYouTube(it.videoId, it.title, it.thumbnailUrl, it.durationSec?.toDouble())
                    },
                    appScope = app.appScope,
                )
            }
        }
    }
}
