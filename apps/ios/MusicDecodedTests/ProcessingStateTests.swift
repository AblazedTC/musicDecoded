import Foundation
import Testing
@testable import MusicDecoded

/// The Processing screen's state. Ports apps/android/.../ui/processing/ProcessingViewModelTest.kt.
struct ProcessingStateTests {
    private func row(_ status: LibraryItem.Status, error: String? = nil, tombstoned: Bool = false) -> ProcessingState.Row {
        ProcessingState.Row(title: "Song", artist: "Artist", status: status, error: error, tombstoned: tombstoned)
    }

    private func job(_ stage: String?, _ progress: Int, status: Job.Status = .processing) -> Job {
        Job(jobId: "j1", trackId: "t1", status: status, stage: stage, progress: progress, error: nil)
    }

    /// Feeds updates in order, like the screen's `onChange`.
    private func fold(_ updates: [(ProcessingState.Row?, Job?)]) -> ProcessingState {
        updates.reduce(ProcessingState()) { state, update in
            ProcessingState.reduce(state, row: update.0, job: update.1)
        }
    }

    /// DESIGN.md §3.5, row by row.
    @Test func stageToStepMatchesTheDesignTable() {
        let table: [String: ProcessingStep] = [
            "fetching_media": .preparing,
            "normalizing_audio": .preparing,
            "detecting_beats": .beats,
            "detecting_chords": .harmony,
            "detecting_key": .harmony,
            "rendering_spectrogram": .timeline,
            "saving_results": .saving,
        ]
        for (stage, step) in table {
            #expect(ProcessingStep(stage: stage) == step, "\(stage)")
        }
        #expect(Set(table.keys) == ProcessingStep.allCases.reduce(into: Set<String>()) { $0.formUnion($1.stages) })
        #expect(ProcessingStep.allCases.map(\.title) == [
            "Preparing audio", "Detecting beats & tempo", "Analyzing harmony", "Generating chord timeline", "Saving transcription",
        ])
        #expect(ProcessingStep(stage: nil) == nil)
        #expect(ProcessingStep(stage: "brand_new_stage") == nil)
    }

    @Test func loadingUntilTheRowExists() {
        #expect(fold([(nil, nil)]).phase == .loading)
    }

    @Test func runningFollowsStageAndProgress() {
        let queued = fold([(row(.analyzing), nil)])
        #expect(queued.phase == .running)
        #expect(queued.step == .preparing)
        #expect(queued.progress == 0)

        let state = fold([(row(.analyzing), nil), (row(.analyzing), job("detecting_chords", 55))])
        #expect(state.step == .harmony)
        #expect(state.progress == 55)
        #expect(state.isDone(.preparing))
        #expect(state.isDone(.beats))
        #expect(state.isActive(.harmony))
        #expect(state.isDone(.timeline) == false)
        #expect(state.title == "Song")
        #expect(state.artist == "Artist")
    }

    @Test func unknownStageKeepsTheStepAndProgressNeverGoesBack() {
        let state = fold([
            (row(.analyzing), job("rendering_spectrogram", 80)),
            (row(.analyzing), job("something_new", 70)),
        ])
        #expect(state.step == .timeline)
        #expect(state.progress == 80)
    }

    @Test func completeJobIsSavingUntilTheRowIsReady() {
        let saving = fold([(row(.analyzing), job(nil, 100, status: .complete))])
        #expect(saving.step == .saving)
        #expect(saving.phase == .running)

        let ready = ProcessingState.reduce(saving, row: row(.ready), job: nil)
        #expect(ready.phase == .complete)
        #expect(ready.progress == 100)
        #expect(ProcessingStep.allCases.allSatisfy(ready.isDone))
    }

    @Test func failedShowsTheErrorAndRetryClearsIt() {
        let failed = fold([(row(.failed, error: "We couldn't decode this audio. Try again."), nil)])
        #expect(failed.phase == .failed)
        #expect(failed.error == "We couldn't decode this audio. Try again.")

        let retried = ProcessingState.reduce(failed, row: row(.analyzing), job: job("fetching_media", 5))
        #expect(retried.phase == .running)
        #expect(retried.error == nil)
        #expect(retried.progress == 5)
    }

    @Test func deletedRowIsLoading() {
        #expect(fold([(row(.analyzing, tombstoned: true), nil)]).phase == .loading)
    }
}
