import Foundation
import MusicDecodedCore
import Testing
@testable import MusicDecoded

/// T-305i: the YouTube bridge, the video clock and the engines that make the video the clock or
/// keep it in step with local audio. Everything runs on fakes; no web view plays anything.
@MainActor
struct VideoSyncTests {
    /// A fake monotonic clock, in seconds.
    final class Clock {
        var now = 100.0
    }

    let clock = Clock()

    private func close(_ a: Double, _ b: Double) -> Bool {
        abs(a - b) < 1e-9
    }

    // MARK: - Bridge messages

    private func message(_ fields: [String: Any]) -> YouTubeBridgeMessage? {
        YouTubeBridgeMessage(body: fields)
    }

    @Test func parsesBridgeMessages() {
        #expect(message(["event": "ready"]) == .ready)
        #expect(message(["event": "state", "state": 1, "time": 12.5]) == .state(.playing, time: 12.5))
        #expect(message(["event": "state", "state": 3]) == .state(.buffering, time: nil))
        #expect(message(["event": "state", "state": -1, "time": 0]) == .state(.unstarted, time: 0))
        #expect(message(["event": "time", "time": 3]) == .time(3))
        #expect(message(["event": "rate", "rate": 0.75]) == .rate(0.75))
        #expect(message(["event": "error", "code": 150]) == .error(150))
        // What WebKit actually delivers: NSNumber values.
        let fromJS: [String: Any] = ["event": "state", "state": NSNumber(value: 2), "time": NSNumber(value: 4.25)]
        #expect(message(fromJS) == .state(.paused, time: 4.25))
    }

    @Test func ignoresUnknownMessages() {
        #expect(YouTubeBridgeMessage(body: "ready") == nil)
        #expect(message(["event": "volume"]) == nil)
        #expect(message(["event": "state", "state": 7]) == nil)
        #expect(message(["event": "time"]) == nil)
        #expect(message(["event": "rate", "rate": 0]) == nil)
    }

    // MARK: - Video clock

    @Test func clockExtrapolatesOnlyWhilePlaying() {
        var videoClock = YouTubeVideoClock()
        videoClock.report(state: .playing, time: 10, at: 100)
        #expect(close(videoClock.position(at: 100.5), 10.5))

        videoClock.setRate(0.5, at: 100.5)
        #expect(close(videoClock.position(at: 101.5), 11))

        videoClock.report(state: .buffering, time: 11, at: 101.5)
        #expect(close(videoClock.position(at: 105), 11))

        videoClock.report(state: .paused, time: 11.2, at: 105)
        #expect(close(videoClock.position(at: 110), 11.2))
    }

    @Test func clockHoldsSmallBackwardJitter() {
        var videoClock = YouTubeVideoClock()
        videoClock.report(state: .playing, time: 10, at: 100)
        #expect(close(videoClock.position(at: 100.5), 10.5))

        // A report 0.1 s behind the extrapolation: the playhead waits instead of jumping back.
        videoClock.report(time: 10.4, at: 100.5)
        #expect(close(videoClock.position(at: 100.5), 10.5))
        #expect(close(videoClock.position(at: 100.7), 10.6))

        // A real jump back (the user scrubbed in the embed) goes through.
        videoClock.report(time: 5, at: 100.7)
        #expect(close(videoClock.position(at: 100.7), 5))
    }

    @Test func clockIgnoresReportsFromBeforeASeek() {
        var videoClock = YouTubeVideoClock()
        videoClock.report(state: .paused, time: 10, at: 99)
        videoClock.seek(to: 60, at: 100)
        #expect(close(videoClock.position(at: 100), 60))

        videoClock.report(time: 10.3, at: 100.1)
        #expect(close(videoClock.position(at: 100.1), 60))

        videoClock.report(time: 60.05, at: 100.3)
        #expect(close(videoClock.position(at: 100.3), 60.05))

        // Stale-report filtering ends with the first matching report, or after the timeout.
        videoClock.seek(to: 30, at: 200)
        videoClock.report(time: 10, at: 202.5)
        #expect(close(videoClock.position(at: 202.5), 10))
    }

    // MARK: - Controller state mapping

    @Test func controllerMapsPlayerStates() {
        let clock = clock
        let controller = YouTubePlayerController(videoId: "dQw4w9WgXcQ", now: { clock.now })
        #expect(controller.isReady == false)

        // Commands before ready only record intent.
        controller.play()
        #expect(controller.isPlaying)
        controller.seek(to: 42)
        #expect(close(controller.currentTime, 42))

        controller.handle(.ready)
        #expect(controller.isReady)

        controller.handle(.state(.playing, time: 42))
        #expect(controller.state == .playing)
        clock.now += 1
        #expect(close(controller.currentTime, 43))

        controller.handle(.state(.paused, time: 43))
        #expect(controller.isPlaying == false)

        controller.handle(.state(.buffering, time: 43))
        #expect(controller.isPlaying)

        // Ended keeps "playing" so a loop can wrap; the player model pauses at the end.
        controller.handle(.state(.ended, time: 200))
        #expect(controller.isPlaying)
        #expect(controller.state == .ended)

        controller.handle(.error(150))
        #expect(controller.hasFailed)
    }

    // MARK: - Video as the clock

    @Test func videoIsTheClockWithoutLocalAudio() {
        let clock = clock
        let video = FakeVideoPlayer()
        let engine = YouTubePlaybackEngine(video: video, durationSec: 10, now: { clock.now })

        video.currentTime = 3.5
        #expect(engine.position == 3.5)
        #expect(engine.videoPlayer === video)

        engine.play()
        #expect(video.playCalls == 1)
        #expect(engine.isPlaying)

        engine.seek(to: 7)
        #expect(video.seeks == [7])

        engine.setSpeed(0.75)
        #expect(video.rates == [0.75])
        #expect(engine.speed == 0.75)

        engine.pause()
        #expect(video.pauseCalls == 1)
        #expect(engine.isPlaying == false)
    }

    @Test func endedVideoReadsAsTheAnalysisLength() {
        let clock = clock
        let video = FakeVideoPlayer()
        let engine = YouTubePlaybackEngine(video: video, durationSec: 10, now: { clock.now })
        video.state = .ended
        video.currentTime = 9.8
        #expect(engine.position == 10)
    }

    @Test func failedVideoFallsBackToASilentClock() {
        let clock = clock
        let video = FakeVideoPlayer()
        let engine = YouTubePlaybackEngine(video: video, durationSec: 10, now: { clock.now })
        video.currentTime = 4
        video.isPlaying = true
        video.hasFailed = true

        #expect(engine.position == 4)
        #expect(engine.isPlaying)
        clock.now += 2
        #expect(close(engine.position, 6))

        engine.pause()
        clock.now += 2
        #expect(close(engine.position, 6))
        #expect(engine.isPlaying == false)
    }

    // MARK: - Video following local audio

    private func followingEngine(_ video: FakeVideoPlayer) -> (VideoFollowingEngine, SilentPlaybackEngine) {
        let clock = clock
        let audio = SilentPlaybackEngine(durationSec: 100, now: { clock.now })
        return (VideoFollowingEngine(audio: audio, video: video, now: { clock.now }), audio)
    }

    @Test func followerIsMutedAndForwardsCommands() {
        let video = FakeVideoPlayer()
        let (engine, audio) = followingEngine(video)
        #expect(video.muted == true)
        #expect(video.rates == [1])

        engine.play()
        #expect(audio.isPlaying)
        #expect(video.isPlaying)
        #expect(engine.isPlaying)

        engine.seek(to: 20)
        #expect(audio.position == 20)
        #expect(video.seeks == [20])

        engine.setSpeed(0.75)
        #expect(audio.speed == 0.75)
        #expect(video.rates == [1, 0.75])

        engine.pause()
        #expect(audio.isPlaying == false)
        #expect(video.isPlaying == false)
    }

    @Test func driftWithin150msIsLeftAlone() {
        let video = FakeVideoPlayer()
        let (engine, audio) = followingEngine(video)
        audio.seek(to: 10)
        audio.play()
        video.isPlaying = true
        video.state = .playing
        video.currentTime = 10.14

        engine.synchronize()
        #expect(video.seeks.isEmpty)

        video.currentTime = 9.86
        engine.synchronize()
        #expect(video.seeks.isEmpty)
    }

    @Test func driftOver150msReseeksTheVideo() {
        let video = FakeVideoPlayer()
        let (engine, audio) = followingEngine(video)
        audio.seek(to: 10)
        audio.play()
        video.isPlaying = true
        video.state = .playing

        video.currentTime = 10.3
        engine.synchronize()
        #expect(video.seeks == [10])

        // Within the cooldown the seek is still landing: no second seek.
        clock.now += 0.5
        video.currentTime = 9.0
        engine.synchronize()
        #expect(video.seeks == [10])

        // After it, the drift is corrected again, to the audio position at that moment.
        clock.now += 0.5
        engine.synchronize()
        #expect(video.seeks.count == 2)
        #expect(close(video.seeks[1], 11))
    }

    @Test func noCorrectionWhileBufferingOrNotReady() {
        let video = FakeVideoPlayer()
        let (engine, audio) = followingEngine(video)
        audio.play()
        video.isPlaying = true
        clock.now += 5
        video.currentTime = 0

        video.state = .buffering
        engine.synchronize()
        #expect(video.seeks.isEmpty)

        video.state = .playing
        video.isReady = false
        engine.synchronize()
        #expect(video.seeks.isEmpty)

        video.isReady = true
        engine.synchronize()
        #expect(video.seeks == [5])
    }

    @Test func videoPlayStateFollowsTheAudio() {
        let video = FakeVideoPlayer()
        let (engine, audio) = followingEngine(video)

        // Audio playing, video stopped on its own: start it again.
        audio.play()
        engine.synchronize()
        #expect(video.playCalls == 1)
        engine.synchronize()
        #expect(video.playCalls == 1)

        // Audio paused (e.g. an interruption), video still playing: pause it.
        audio.pause()
        engine.synchronize()
        #expect(video.pauseCalls == 1)
    }

    // MARK: - Player model

    private func makeModel(_ engine: any PlaybackEngine) async throws -> PlayerModel {
        let analysis = try PlayerFixtures.synthetic(source: .youtube, sourceRef: "dQw4w9WgXcQ")
        let track = PlayerTrack(analysis: analysis, title: "Song", artist: nil, favorite: false, audioURL: nil)
        let model = PlayerModel(
            loadTrack: { track },
            voicings: nil,
            makeEngine: { _ in engine },
            tickInterval: nil
        )
        await model.load()
        return model
    }

    @Test func scrubbingTheTimelineSeeksTheVideo() async throws {
        let clock = clock
        let video = FakeVideoPlayer()
        let engine = YouTubePlaybackEngine(video: video, durationSec: 10, now: { clock.now })
        let model = try await makeModel(engine)
        #expect(model.videoPlayer === video)

        model.seek(to: 5)
        #expect(video.seeks == [5])

        model.togglePlay()
        #expect(video.playCalls == 1)
        video.currentTime = 6.5
        model.tick()
        #expect(model.position == 6.5)
        #expect(model.chordIndex == 3)
    }

    @Test func tickKeepsTheFollowingVideoInStep() async throws {
        let clock = clock
        let video = FakeVideoPlayer()
        let audio = SilentPlaybackEngine(durationSec: 10, now: { clock.now })
        let model = try await makeModel(VideoFollowingEngine(audio: audio, video: video, now: { clock.now }))

        model.togglePlay()
        video.state = .playing
        clock.now += 2
        video.currentTime = 1.5
        model.tick()

        #expect(close(model.position, 2))
        #expect(video.seeks.count == 1)
        #expect(close(video.seeks[0], 2))
    }
}
