package com.musicdecoded.mobile.sources.soundcloud

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.library.TranscriptionRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SoundCloudUiState(
    val text: String = "",
    /** The text doesn't contain a track link (shown under the field). */
    val linkError: String? = null,
    /** `POST /jobs` failed (offline, `feature_disabled`, rate limit…): shown with Retry. */
    val error: String? = null,
    val starting: Boolean = false,
    /** `false` once the server config says SoundCloud is off. */
    val enabled: Boolean = true,
    /** One-shot: the job exists, open Processing for this trackId. */
    val startedTrackId: String? = null,
)

/** Paste / share-sheet a SoundCloud link → `soundcloud` job (T-502a). */
class SoundCloudViewModel(
    /** Pasted or shared text; may be a whole sentence around the link. */
    initialText: String,
    /** `ClientConfig.soundcloudEnabled`, `null` while unknown (the server still decides). */
    enabled: Boolean?,
    private val startJob: suspend (url: String, title: String) -> String,
    /** Outlives the dialog, so dismissing it doesn't cancel a request in flight. */
    private val appScope: CoroutineScope,
) : ViewModel() {
    private val _state = MutableStateFlow(initialState(initialText, enabled))
    val state: StateFlow<SoundCloudUiState> = _state.asStateFlow()

    fun onTextChange(text: String) = _state.update { it.copy(text = text, linkError = null, error = null) }

    fun decode() {
        val current = _state.value
        if (current.starting || !current.enabled) return
        val url = SoundCloudUrl.extract(current.text)
            ?: return _state.update { it.copy(linkError = NO_LINK) }
        _state.update { it.copy(text = url, starting = true, linkError = null, error = null) }
        appScope.launch {
            try {
                val trackId = startJob(url, SoundCloudUrl.titleFrom(url))
                _state.update { it.copy(starting = false, startedTrackId = trackId) }
            } catch (e: TranscriptionException) {
                val message = e.message ?: TranscriptionRepository.GENERIC
                _state.update {
                    it.copy(starting = false, error = message, enabled = message != TranscriptionRepository.SOUNDCLOUD_DISABLED)
                }
            }
        }
    }

    fun navigated() = _state.update { it.copy(startedTrackId = null) }

    companion object {
        const val NO_LINK = "Paste a link to a SoundCloud track (soundcloud.com/artist/track)."

        private fun initialState(text: String, enabled: Boolean?): SoundCloudUiState {
            val disabled = enabled == false
            if (text.isBlank()) {
                return SoundCloudUiState(enabled = !disabled, error = TranscriptionRepository.SOUNDCLOUD_DISABLED.takeIf { disabled })
            }
            val url = SoundCloudUrl.extract(text)
            return SoundCloudUiState(
                text = url ?: text.trim(),
                linkError = NO_LINK.takeIf { url == null },
                enabled = !disabled,
                error = TranscriptionRepository.SOUNDCLOUD_DISABLED.takeIf { disabled },
            )
        }

        fun factory(app: MusicDecodedApplication, initialText: String): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                SoundCloudViewModel(
                    initialText,
                    enabled = app.authRepository.config.value?.soundcloudEnabled,
                    startJob = { url, title -> app.transcriptionRepository.startSoundCloud(url, title) },
                    appScope = app.appScope,
                )
            }
        }
    }
}
