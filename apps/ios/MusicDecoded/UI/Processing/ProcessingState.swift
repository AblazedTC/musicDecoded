import Foundation

/// The five steps of DESIGN.md §3.5 and the `Job.stage` values each one covers (exactly the table).
enum ProcessingStep: Int, CaseIterable, Comparable, Sendable {
    case preparing, beats, harmony, timeline, saving

    var title: String {
        switch self {
        case .preparing: "Preparing audio"
        case .beats: "Detecting beats & tempo"
        case .harmony: "Analyzing harmony"
        case .timeline: "Generating chord timeline"
        case .saving: "Saving transcription"
        }
    }

    var stages: Set<String> {
        switch self {
        case .preparing: ["fetching_media", "normalizing_audio"]
        case .beats: ["detecting_beats"]
        case .harmony: ["detecting_chords", "detecting_key"]
        case .timeline: ["rendering_spectrogram"]
        case .saving: ["saving_results"]
        }
    }

    /// `nil` for a missing or unknown stage (the screen then keeps the step it was on).
    init?(stage: String?) {
        guard let stage, let step = Self.allCases.first(where: { $0.stages.contains(stage) }) else { return nil }
        self = step
    }

    static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// What the Processing screen shows for one track. Folded from the library row and its job, so an
/// unknown stage keeps the last step and the progress never moves backwards.
struct ProcessingState: Equatable, Sendable {
    enum Phase: Sendable {
        case loading, running, complete, failed
    }

    /// The library row fields the screen reads.
    struct Row: Equatable, Sendable {
        var title: String
        var artist: String?
        var status: LibraryItem.Status
        var error: String?
        var tombstoned: Bool

        init(title: String, artist: String? = nil, status: LibraryItem.Status, error: String? = nil, tombstoned: Bool = false) {
            self.title = title
            self.artist = artist
            self.status = status
            self.error = error
            self.tombstoned = tombstoned
        }

        init(_ item: LibraryItem) {
            self.init(title: item.title, artist: item.artist, status: item.status, error: item.error, tombstoned: item.tombstoned)
        }
    }

    var phase = Phase.loading
    var title = ""
    var artist: String?
    /// The step in progress (all earlier ones are done).
    var step = ProcessingStep.preparing
    /// Overall 0–100, from `Job.progress`.
    var progress = 0
    var error: String?

    func isDone(_ step: ProcessingStep) -> Bool {
        phase == .complete || step < self.step
    }

    func isActive(_ step: ProcessingStep) -> Bool {
        phase == .running && step == self.step
    }

    /// Folds the latest row and job into `previous`.
    static func reduce(_ previous: ProcessingState, row: Row?, job: Job?) -> ProcessingState {
        guard let row, row.tombstoned == false else {
            var next = previous
            next.phase = .loading
            return next
        }
        var next = previous
        next.title = row.title
        next.artist = row.artist
        next.error = nil
        switch row.status {
        case .ready:
            next.phase = .complete
            next.step = .saving
            next.progress = 100
        case .failed:
            next.phase = .failed
            next.error = row.error
        case .analyzing:
            let wasRunning = previous.phase == .running
            // A finished job is still being downloaded and saved until the row turns ready.
            if job?.status == .complete {
                next.step = .saving
            } else if let step = ProcessingStep(stage: job?.stage) {
                next.step = step
            } else if wasRunning == false {
                next.step = .preparing
            }
            let progress = min(max(job?.progress ?? 0, 0), 100)
            // Never moves backwards while the same screen watches one run.
            next.progress = wasRunning ? max(previous.progress, progress) : progress
            next.phase = .running
        }
        return next
    }
}
