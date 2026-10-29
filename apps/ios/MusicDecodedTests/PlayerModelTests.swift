import Foundation
import MusicDecodedCore
import Testing
@testable import MusicDecoded

/// Player state logic against `SilentPlaybackEngine` driven by a fake clock (no AVPlayer).
/// Ports apps/android/.../player/PlayerViewModelTest.kt.
@MainActor
struct PlayerModelTests {
    /// Fake monotonic clock, in seconds, and what the model saved.
    final class Harness {
        var now = 0.0
        var engine: SilentPlaybackEngine?
        var savedFavorites: [Bool] = []

        func advance(_ seconds: Double) {
            now += seconds
        }
    }

    let harness = Harness()
    let voicings: ChordVoicings

    init() throws {
        voicings = try PlayerFixtures.voicings()
    }

    /// Synthetic: A:maj 0–2 · N 2–4 · N 4–5 · D:maj 5–8 · N 8–10; beats every 1 s, downbeats 0, 4, 8.
    private func makeModel(analysis: Analysis? = nil, hasTrack: Bool = true) async throws -> PlayerModel {
        let resolved = try analysis ?? PlayerFixtures.synthetic()
        let harness = harness
        let track = hasTrack ? PlayerTrack(analysis: resolved, title: "Song", artist: "Artist", favorite: false, audioURL: nil) : nil
        let model = PlayerModel(
            loadTrack: { track },
            voicings: voicings,
            makeEngine: { track in
                let engine = SilentPlaybackEngine(durationSec: track.analysis.meta.durationSec, now: { harness.now })
                harness.engine = engine
                return engine
            },
            saveFavorite: { harness.savedFavorites.append($0) },
            tickInterval: nil
        )
        await model.load()
        return model
    }

    @Test func loadsTheTrack() async throws {
        let model = try await makeModel()
        #expect(model.isLoading == false)
        #expect(model.track?.title == "Song")
        #expect(model.position == 0)
        #expect(model.chordIndex == 0)
    }

    @Test func missingTrackStopsLoadingWithoutATrack() async throws {
        let model = try await makeModel(hasTrack: false)
        #expect(model.isLoading == false)
        #expect(model.track == nil)
        model.togglePlay() // no engine: no crash
        #expect(model.isPlaying == false)
    }

    @Test func engineClockDrivesThePosition() async throws {
        let model = try await makeModel()
        model.togglePlay()
        #expect(model.isPlaying)
        harness.advance(1.5)
        model.tick()
        #expect(abs(model.position - 1.5) < 1e-6)
        model.togglePlay()
        harness.advance(3)
        model.tick()
        #expect(model.isPlaying == false)
        #expect(abs(model.position - 1.5) < 1e-6)
    }

    @Test func speedScalesTheClockAndSnapsToSteps() async throws {
        let model = try await makeModel()
        model.setSpeed(0.77)
        #expect(model.speed == 0.75)
        model.togglePlay()
        harness.advance(2)
        model.tick()
        #expect(abs(model.position - 1.5) < 1e-6)
        model.setSpeed(3)
        #expect(model.speed == 1.25)
        model.setSpeed(0.1)
        #expect(model.speed == 0.5)
    }

    @Test func speedTapTogglesBetweenNormalAndPractice() async throws {
        let model = try await makeModel()
        model.toggleSpeed()
        #expect(model.speed == 0.75)
        model.toggleSpeed()
        #expect(model.speed == 1)
        model.setSpeed(1.1) // any non-1× speed taps back to 1×
        model.toggleSpeed()
        #expect(model.speed == 1)
    }

    @Test func speedStepsCover050to125() {
        #expect(PlayerModel.speedSteps.count == 16)
        #expect(PlayerModel.speedSteps.first == 0.5)
        #expect(PlayerModel.speedSteps.last == 1.25)
    }

    @Test func seekIsClampedToTheTrack() async throws {
        let model = try await makeModel()
        model.seek(to: -3)
        #expect(model.position == 0)
        model.seek(to: 99)
        #expect(model.position == 10)
        #expect(harness.engine?.position == 10)
    }

    @Test func nextAndPreviousChordSkipNoChord() async throws {
        let model = try await makeModel()
        model.seek(to: 0.5)
        model.nextChord()
        #expect(model.position == 5) // skips the N runs
        model.seek(to: 6.5) // > 1 s into D:maj → restart it
        model.previousChord()
        #expect(model.position == 5)
        model.previousChord() // at its start → previous real chord
        #expect(model.position == 0)
    }

    @Test func loopButtonLoopsTheCurrentBar() async throws {
        let model = try await makeModel()
        model.seek(to: 5.5)
        model.toggleLoop()
        #expect(model.loop == 4..<8)
        model.togglePlay()
        harness.advance(2.6) // 5.5 + 2.6 = 8.1 → wraps to the bar start
        model.tick()
        #expect(abs(model.position - 4) < 1e-6)
        harness.advance(1)
        model.tick()
        #expect(abs(model.position - 5) < 1e-6)
        model.toggleLoop()
        #expect(model.loop == nil)
    }

    @Test func customLoopJumpsInWhenThePlayheadIsOutside() async throws {
        let model = try await makeModel()
        model.seek(to: 0.5)
        model.setLoop(2..<4)
        #expect(model.position == 2)
        model.clearLoop()
        #expect(model.loop == nil)
        model.seek(to: 3)
        model.setLoop(2..<6) // already inside: stays
        #expect(model.position == 3)
    }

    @Test func stopsAtTheEndAndRestartsFromTheTop() async throws {
        let model = try await makeModel()
        model.seek(to: 9)
        model.togglePlay()
        harness.advance(5)
        model.tick()
        #expect(model.isPlaying == false)
        #expect(abs(model.position - 10) < 1e-6)
        model.togglePlay()
        #expect(model.isPlaying)
        #expect(abs(model.position) < 1e-6)
    }

    @Test func pauseOnlyPauses() async throws {
        let model = try await makeModel()
        model.pause()
        #expect(model.isPlaying == false)
        model.togglePlay()
        model.pause()
        #expect(model.isPlaying == false)
    }

    @Test func transposeIsClampedAndChangesNamesAndNotes() async throws {
        let model = try await makeModel()
        model.setTranspose(9)
        #expect(model.transpose == 6)
        model.setTranspose(-9)
        #expect(model.transpose == -6)
        model.setTranspose(3)
        let chords = model.chords
        #expect(chords.name("A:maj") == "C")
        #expect(chords.notes("A:maj") == voicings.notes(for: "C:maj"))
        #expect(chords.noteNames("A:maj") == "C · E · G")
    }

    @Test func basicChordsSimplifyNamesAndNotes() async throws {
        let model = try await makeModel()
        #expect(model.chords.name("F#:min7") == "F♯m7")
        model.toggleChordsMode()
        #expect(model.chordsMode == .basic)
        let chords = model.chords
        #expect(chords.name("F#:min7") == "F♯m")
        #expect(chords.notes("F#:min7") == voicings.notes(for: "F#:min"))
        #expect(chords.bassNote("D:maj/3") == nil)
    }

    @Test func slashBassIsTheLowestNote() async throws {
        let chords = try await makeModel().chords
        let notes = chords.notes("D:maj/3")
        #expect(chords.bassNote("D:maj/3") == notes.first)
        #expect(notes[0] < notes[1])
        #expect(chords.bassNote("D:maj") == nil)
    }

    @Test func favoriteTogglesAndSaves() async throws {
        let model = try await makeModel()
        model.toggleFavorite()
        #expect(model.favorite)
        model.toggleFavorite()
        #expect(harness.savedFavorites == [true, false])
    }

    @Test func videoOnlyForYoutubeSources() async throws {
        let model = try await makeModel()
        model.selectVisualizer(.video)
        #expect(model.visualizer == .piano)

        let youtube = try PlayerFixtures.synthetic(source: .youtube, sourceRef: "abc")
        let youtubeModel = try await makeModel(analysis: youtube)
        #expect(youtubeModel.videoAvailable)
        youtubeModel.selectVisualizer(.video)
        #expect(youtubeModel.visualizer == .video)
    }

    @Test func noteNamesToggle() async throws {
        let model = try await makeModel()
        #expect(model.showNoteNames)
        model.toggleNoteNames()
        #expect(model.showNoteNames == false)
    }

    @Test func chordAndUpNextFollowThePlayhead() async throws {
        let model = try await makeModel()
        model.seek(to: 0.5)
        #expect(model.currentLabel == "A:maj")
        #expect(model.upcoming?.chordIndex == 3)
        #expect(model.upcoming?.inBeats == 5) // beats 1…5 until D:maj at 5
        model.seek(to: 6)
        #expect(model.currentLabel == "D:maj")
        #expect(model.upcoming == nil)
    }

    @Test func formatsChipValues() {
        #expect(PlayerFormat.transpose(2) == "+2")
        #expect(PlayerFormat.transpose(-3) == "−3")
        #expect(PlayerFormat.transpose(0) == "0")
        #expect(PlayerFormat.speed(0.75) == "0.75×")
        #expect(PlayerFormat.speed(1) == "1.00×")
        #expect(PlayerFormat.speed(1.25) == "1.25×")
    }
}
