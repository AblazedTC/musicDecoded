package com.musicdecoded.mobile.sources.youtube

import android.os.Bundle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.TranscriptionException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class YouTubeBrowserUiState(
    /** The page being shown is a single video. */
    val videoId: String? = null,
    val canGoBack: Boolean = false,
    val canGoForward: Boolean = false,
    /** `POST /jobs` in flight. */
    val starting: Boolean = false,
    val error: String? = null,
    /** One-shot: the job exists, open Processing for this trackId. */
    val startedTrackId: String? = null,
)

/** In-app m.youtube.com (T-308a): "Transcribe this" on a video page → `youtube` job → Processing. */
class YouTubeBrowserViewModel(
    private val startJob: suspend (videoId: String, title: String) -> String,
    /** Outlives this screen, so leaving it doesn't cancel a request in flight. */
    private val appScope: CoroutineScope,
) : ViewModel() {
    private val _state = MutableStateFlow(YouTubeBrowserUiState())
    val state: StateFlow<YouTubeBrowserUiState> = _state.asStateFlow()

    /** The WebView's history while the screen is off the composition (e.g. under Processing). */
    var webState: Bundle? = null

    /** The current page's title, from `onReceivedTitle`; cleared on every navigation so a stale one is never used. */
    private var pageTitle: String? = null

    fun titleReceived(title: String?) {
        pageTitle = title
    }

    fun pageChanged(url: String?, canGoBack: Boolean, canGoForward: Boolean) = _state.update {
        pageTitle = null
        val videoId = url?.let(YouTubeUrl::videoId)
        it.copy(
            videoId = videoId,
            canGoBack = canGoBack,
            canGoForward = canGoForward,
            error = it.error.takeIf { _ -> videoId == it.videoId },
        )
    }

    fun transcribe() {
        val videoId = _state.value.videoId ?: return
        if (_state.value.starting) return
        _state.update { it.copy(starting = true, error = null) }
        appScope.launch {
            try {
                val trackId = startJob(videoId, YouTubeUrl.pageTitle(pageTitle) ?: LibraryItem.YOUTUBE_PLACEHOLDER_TITLE)
                _state.update { it.copy(starting = false, startedTrackId = trackId) }
            } catch (e: TranscriptionException) {
                _state.update { it.copy(starting = false, error = e.message) }
            }
        }
    }

    fun navigated() = _state.update { it.copy(startedTrackId = null) }

    companion object {
        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                YouTubeBrowserViewModel(
                    startJob = { videoId, title ->
                        val thumb = YouTubeSearchViewModel.pastedVideo(videoId).thumbnailUrl
                        app.transcriptionRepository.startYouTube(videoId, title, thumb, null)
                    },
                    appScope = app.appScope,
                )
            }
        }
    }
}
