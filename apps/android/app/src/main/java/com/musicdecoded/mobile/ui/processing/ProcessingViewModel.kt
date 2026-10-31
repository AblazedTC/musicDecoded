package com.musicdecoded.mobile.ui.processing

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.network.JobStatus
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.scan
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/** The five steps of DESIGN.md §3.5 and the `Job.stage` values each one covers (exactly the table). */
enum class ProcessingStep(val label: String, val stages: Set<String>) {
    PREPARING("Preparing audio", setOf("fetching_media", "normalizing_audio")),
    BEATS("Detecting beats & tempo", setOf("detecting_beats")),
    HARMONY("Analyzing harmony", setOf("detecting_chords", "detecting_key")),
    TIMELINE("Generating chord timeline", setOf("rendering_spectrogram")),
    SAVING("Saving transcription", setOf("saving_results")),
    ;

    companion object {
        /** `null` for a missing or unknown stage (the screen then keeps the step it was on). */
        fun forStage(stage: String?): ProcessingStep? = entries.firstOrNull { stage in it.stages }
    }
}

enum class ProcessingPhase { LOADING, RUNNING, COMPLETE, FAILED }

data class ProcessingUiState(
    val phase: ProcessingPhase = ProcessingPhase.LOADING,
    val title: String = "",
    val artist: String? = null,
    /** The step in progress (all earlier ones are done). */
    val step: ProcessingStep = ProcessingStep.PREPARING,
    /** Overall 0–100, from `Job.progress`. */
    val progress: Int = 0,
    val error: String? = null,
) {
    fun isDone(step: ProcessingStep): Boolean = phase == ProcessingPhase.COMPLETE || step < this.step
    fun isActive(step: ProcessingStep): Boolean = phase == ProcessingPhase.RUNNING && step == this.step

    companion object {
        /** Folds the latest row and job into the previous state (so an unknown stage keeps the last step). */
        fun reduce(previous: ProcessingUiState, item: LibraryItem?, job: Job?): ProcessingUiState {
            if (item == null || item.deleted) return previous.copy(phase = ProcessingPhase.LOADING)
            val base = previous.copy(title = item.title, artist = item.artist, error = null)
            return when (item.status) {
                LibraryStatus.READY -> base.copy(phase = ProcessingPhase.COMPLETE, step = ProcessingStep.SAVING, progress = 100)
                LibraryStatus.FAILED -> base.copy(phase = ProcessingPhase.FAILED, error = item.error)
                LibraryStatus.ANALYZING -> {
                    // A finished job is still being downloaded and saved until the row turns ready.
                    val step = when {
                        job?.status == JobStatus.COMPLETE -> ProcessingStep.SAVING
                        else -> ProcessingStep.forStage(job?.stage)
                            ?: if (previous.phase == ProcessingPhase.RUNNING) previous.step else ProcessingStep.PREPARING
                    }
                    val progress = (job?.progress ?: 0).coerceIn(0, 100)
                    base.copy(
                        phase = ProcessingPhase.RUNNING,
                        step = step,
                        // Never moves backwards while the same screen watches one run.
                        progress = if (previous.phase == ProcessingPhase.RUNNING) maxOf(previous.progress, progress) else progress,
                    )
                }
            }
        }
    }
}

/** DESIGN.md §3.5. Watches one track's row and job; leaving the screen never cancels the job. */
class ProcessingViewModel(
    private val trackId: String,
    item: Flow<LibraryItem?>,
    jobs: StateFlow<Map<String, Job>>,
    private val retry: suspend (trackId: String) -> Unit,
) : ViewModel() {
    val state: StateFlow<ProcessingUiState> = combine(item, jobs.map { it[trackId] }) { row, job -> row to job }
        .scan(ProcessingUiState()) { previous, (row, job) -> ProcessingUiState.reduce(previous, row, job) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), ProcessingUiState())

    /** "Try again" after a failure. A failure to even start shows up as the row's error. */
    fun retry() {
        viewModelScope.launch {
            try {
                retry(trackId)
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // TranscriptionRepository.retry already marked the row failed with a user-facing message.
            }
        }
    }

    companion object {
        fun factory(app: MusicDecodedApplication, trackId: String): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val repository: TranscriptionRepository = app.transcriptionRepository
                ProcessingViewModel(trackId, app.libraryDatabase.libraryDao().observe(trackId), repository.jobs, repository::retry)
            }
        }
    }
}
