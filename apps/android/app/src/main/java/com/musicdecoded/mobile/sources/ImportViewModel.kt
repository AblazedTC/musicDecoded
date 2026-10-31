package com.musicdecoded.mobile.sources

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LocalAudio
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.library.TranscriptionRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Where the user picks the file from (Home source cards). */
enum class ImportFrom { FILES, MEDIA_LIBRARY }

data class ImportUiState(
    /** The system picker should be opened (on entry and after "Choose another file"). */
    val pickerPending: Boolean = true,
    val copying: Boolean = false,
    val draft: ImportDraft? = null,
    /** Upload + job request in progress. */
    val starting: Boolean = false,
    val error: String? = null,
    /** Set when the flow is over: the trackId if decoding started, `null` if the user backed out. */
    val finished: Finished? = null,
) {
    data class Finished(val trackId: String?)
}

/** Files / Media Library import (T-205a): pick → copy → Import sheet → [TranscriptionRepository.start]. */
class ImportViewModel(
    private val importer: AudioImporter,
    private val repository: TranscriptionRepository,
    /** Outlives this screen, so leaving it doesn't cancel an upload in flight. */
    private val appScope: CoroutineScope,
) : ViewModel() {
    private val _state = MutableStateFlow(ImportUiState())
    val state: StateFlow<ImportUiState> = _state.asStateFlow()

    fun onPickerLaunched() = _state.update { it.copy(pickerPending = false) }

    fun onPicked(uri: Uri?) {
        if (uri == null) {
            // Cancelled the picker: back out, unless a sheet is still showing an earlier file.
            if (_state.value.draft == null) finish(null)
            return
        }
        _state.update { it.copy(copying = true, error = null) }
        viewModelScope.launch {
            try {
                val draft = importer.import(uri)
                _state.update { it.copy(copying = false, draft = draft) }
            } catch (e: TranscriptionException) {
                _state.update { it.copy(copying = false, error = e.message) }
            }
        }
    }

    fun importAndDecode() {
        val draft = _state.value.draft ?: return
        if (_state.value.starting) return
        _state.update { it.copy(starting = true, error = null) }
        appScope.launch {
            try {
                val trackId = repository.start(LocalAudio(draft.file, draft.title, Source.FILE, draft.durationSec))
                _state.update { it.copy(starting = false, draft = null, finished = ImportUiState.Finished(trackId)) }
            } catch (e: TranscriptionException) {
                _state.update { it.copy(starting = false, error = e.message) }
            }
        }
    }

    fun chooseAnother() {
        if (_state.value.starting) return
        discardDraft()
        _state.update { it.copy(pickerPending = true, error = null) }
    }

    fun dismiss() {
        if (_state.value.starting) return
        discardDraft()
        finish(null)
    }

    override fun onCleared() {
        if (!_state.value.starting) discardDraft()
    }

    private fun finish(trackId: String?) = _state.update { it.copy(finished = ImportUiState.Finished(trackId)) }

    private fun discardDraft() {
        _state.value.draft?.file?.delete()
        _state.update { it.copy(draft = null) }
    }
}
