import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

/// VoiceOver text builders and the shared error state (T-503i).
@MainActor
struct AccessibilityTests {
    let container: ModelContainer

    init() throws {
        container = try ModelContainer(for: LibraryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
    }

    private func item(status: LibraryItem.Status = .ready) -> LibraryItem {
        let item = LibraryItem(trackId: "a", title: "Sunny Days", artist: "Fixture", source: .file, durationSec: 263.4, status: status)
        item.key = "F# minor"
        item.bpm = 136.1
        item.localAudioPath = "a.m4a"
        container.mainContext.insert(item)
        return item
    }

    @Test func readyRowReadsMetadataInWords() {
        let row = item()
        row.favorite = true
        #expect(
            SongRowState(row, job: nil).accessibilityLabel
                == "Sunny Days, Fixture, 4:23, F sharp minor, 136 BPM, available offline, favorite"
        )
    }

    @Test func analyzingAndFailedRowsReadTheirStatus() {
        let analyzing = item(status: .analyzing)
        let job = Job(jobId: "j1", trackId: "a", status: .processing, stage: nil, progress: 42, error: nil)
        #expect(SongRowState(analyzing, job: job).accessibilityLabel == "Sunny Days, Fixture, analyzing, 42 percent")

        analyzing.status = .failed
        #expect(SongRowState(analyzing, job: nil).accessibilityLabel == "Sunny Days, Fixture, failed")
    }

    @Test func errorStateTitles() {
        #expect(ErrorState.title(for: Transcription.offlineMessage) == "You're offline")
        #expect(ErrorState.title(for: Transcription.rateLimitedMessage) == "Something went wrong")
    }

    @Test func seekBarMapsFractions() {
        #expect(SeekBar.fraction(50, in: 0...200) == 0.25)
        #expect(SeekBar.fraction(-5, in: 0...200) == 0)
        #expect(SeekBar.fraction(500, in: 0...200) == 1)
        #expect(SeekBar.fraction(3, in: 3...3) == 0)
        #expect(SeekBar.value(atFraction: 0.5, in: 0...200) == 100)
        #expect(SeekBar.value(atFraction: 2, in: 0...200) == 200)
    }
}
