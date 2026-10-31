import AVFoundation
import Foundation
import Observation

/// Record source (T-206i): AVAudioRecorder → title → `TranscriptionService.start`. Max 10 minutes.
@MainActor
@Observable
final class RecorderModel {
    enum Phase {
        /// Microphone ready, nothing recorded yet.
        case idle
        case recording
        case paused
        /// Recording finished and kept; waiting for a title (or the prompt was dismissed).
        case stopped
        /// Uploading + requesting the job (and after the job started).
        case starting
    }

    enum Permission {
        case undetermined, granted, denied
    }

    static let maxDuration: TimeInterval = 10 * 60

    private(set) var phase = Phase.idle
    private(set) var permission: Permission
    /// Recorded time, not counting pauses.
    private(set) var elapsed: TimeInterval = 0
    private(set) var levelDb = InputLevel.minDb
    /// Recent levels, 0-1, oldest first.
    private(set) var waveform: [Float] = []
    /// Drives the title prompt.
    var askTitle = false
    private(set) var defaultTitle = ""
    var errorMessage: String?
    /// Set once the job has started.
    private(set) var startedTrackId: String?

    @ObservationIgnored private var recorder: AVAudioRecorder?
    @ObservationIgnored private var fileURL: URL?
    @ObservationIgnored private var ticker: Task<Void, Never>?

    var hasRecording: Bool { phase != .idle }

    init() {
        permission = Self.currentPermission()
    }

    // MARK: - Permission

    func requestPermission() async {
        guard Self.currentPermission() == .undetermined else {
            permission = Self.currentPermission()
            return
        }
        let granted = await AVAudioApplication.requestRecordPermission()
        permission = granted ? .granted : .denied
    }

    /// Back from Settings: pick up a permission changed there.
    func refreshPermission() {
        permission = Self.currentPermission()
    }

    private static func currentPermission() -> Permission {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: .granted
        case .denied: .denied
        default: .undetermined
        }
    }

    // MARK: - Actions

    /// The big button: start, pause or resume.
    func toggleRecord(using service: TranscriptionService) {
        switch phase {
        case .idle: startRecording(using: service)
        case .recording: pause()
        case .paused: resume()
        case .stopped, .starting: break
        }
    }

    /// Finish ✓ / Save: stop the recording and ask for a title.
    func finish() {
        switch phase {
        case .recording, .paused:
            stopTicker()
            let duration = max(recorder?.currentTime ?? 0, elapsed)
            recorder?.stop()
            recorder = nil
            deactivateSession()
            guard duration >= 1 else {
                discard()
                errorMessage = "That recording was too short. Try again."
                return
            }
            elapsed = duration
            levelDb = InputLevel.minDb
            phase = .stopped
            defaultTitle = "Recording " + Date.now.formatted(.dateTime.month(.abbreviated).day().hour().minute())
            askTitle = true
        case .stopped:
            askTitle = true
        case .idle, .starting:
            break
        }
    }

    func discard() {
        guard phase != .starting else { return }
        stopTicker()
        if let recorder {
            recorder.stop()
            deactivateSession()
        }
        recorder = nil
        if let fileURL {
            try? FileManager.default.removeItem(at: fileURL)
        }
        fileURL = nil
        phase = .idle
        elapsed = 0
        levelDb = InputLevel.minDb
        waveform = []
        askTitle = false
        errorMessage = nil
    }

    /// Uploads the recording under `title` (or the default title) and starts the job.
    func save(title: String, using service: TranscriptionService) async {
        guard phase == .stopped, let fileURL else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = trimmed.isEmpty ? defaultTitle : trimmed
        phase = .starting
        askTitle = false
        errorMessage = nil
        do {
            let audio = LocalAudio(fileURL: fileURL, title: name, source: .recording, durationSec: elapsed)
            let trackId = try await service.start(audio)
            self.fileURL = nil
            startedTrackId = trackId
        } catch {
            phase = .stopped
            errorMessage = Transcription.userMessage(for: error)
        }
    }

    /// Background apps can't use the mic, so pause when the app leaves the foreground.
    func onBackgrounded() {
        if phase == .recording {
            pause()
        }
    }

    /// Leaving the screen throws the recording away, unless it's already uploading.
    func close() {
        discard()
    }

    // MARK: - Recording

    /// AAC, 44.1 kHz, mono, in an `.m4a` (PLAN.md §3.2).
    private static var settings: [String: Any] {
        [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 44_100.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 128_000,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]
    }

    private func startRecording(using service: TranscriptionService) {
        errorMessage = nil
        var url: URL?
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .default)
            try session.setActive(true)
            let target = try service.newPendingFileURL(extension: "m4a")
            url = target
            let recorder = try AVAudioRecorder(url: target, settings: Self.settings)
            recorder.isMeteringEnabled = true
            guard recorder.record() else { throw TranscriptionError(message: "") }
            self.recorder = recorder
        } catch {
            recorder = nil
            deactivateSession()
            if let url {
                try? FileManager.default.removeItem(at: url)
            }
            errorMessage = "Couldn't start the microphone. Close other apps using it and try again."
            return
        }
        fileURL = url
        elapsed = 0
        waveform = []
        phase = .recording
        startTicker()
    }

    private func pause() {
        recorder?.pause()
        stopTicker()
        levelDb = InputLevel.minDb
        phase = .paused
    }

    private func resume() {
        try? AVAudioSession.sharedInstance().setActive(true)
        guard recorder?.record() == true else {
            errorMessage = "Couldn't resume the microphone. Finish or discard this recording."
            return
        }
        errorMessage = nil
        phase = .recording
        startTicker()
    }

    private func deactivateSession() {
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startTicker() {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.tick()
                try? await Task.sleep(for: .milliseconds(60))
            }
        }
    }

    private func stopTicker() {
        ticker?.cancel()
        ticker = nil
    }

    private func tick() {
        guard let recorder, phase == .recording else { return }
        // An interruption (call, Siri, another app's recording) pauses the recorder underneath us.
        guard recorder.isRecording else {
            pause()
            return
        }
        recorder.updateMeters()
        let db = InputLevel.clamp(recorder.peakPower(forChannel: 0))
        elapsed = recorder.currentTime
        levelDb = db
        waveform.append(InputLevel.normalized(db))
        if waveform.count > InputLevel.waveformBars {
            waveform.removeFirst(waveform.count - InputLevel.waveformBars)
        }
        if elapsed >= Self.maxDuration {
            finish()
        }
    }
}
