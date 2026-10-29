import Foundation
import MusicDecodedCore
import Observation

/// Everything the player needs about one track. `audioURL` is `nil` when the audio isn't on this device.
struct PlayerTrack: Sendable {
    let analysis: Analysis
    let title: String
    let artist: String?
    let favorite: Bool
    let audioURL: URL?
}

enum ChordsMode: Sendable {
    case basic, full
}

enum Visualizer: String, CaseIterable, Hashable, Sendable {
    case piano, video

    var title: String {
        switch self {
        case .piano: "Piano"
        case .video: "Video"
        }
    }
}

/// State for the Player and the Piano learning view (the piano view shares the player's instance).
/// The `PlaybackEngine` position is the clock: while playing, a ~60 Hz loop calls `tick()`, which
/// publishes `position` and enforces the loop. The timeline reads `livePosition` every frame.
@MainActor
@Observable
final class PlayerModel {
    typealias TrackLoader = @MainActor () async -> PlayerTrack?
    typealias EngineFactory = @MainActor (PlayerTrack) -> any PlaybackEngine
    typealias FavoriteSaver = @MainActor (Bool) -> Void

    static let minSpeed = 0.5
    static let maxSpeed = 1.25
    static let practiceSpeed = 0.75
    static let maxTranspose = 6
    /// Long-press speed choices: 0.50…1.25 in 0.05 steps.
    static let speedSteps: [Double] = (10...25).map { Double($0) / 20 }

    /// A position this close to the end counts as "ended", so play restarts from the top.
    private static let endTolerance = 0.05
    /// Seeking slightly before a loop (e.g. snapping) doesn't count as leaving it.
    private static let loopEntryTolerance = 0.25

    private(set) var isLoading = true
    private(set) var track: PlayerTrack?
    private(set) var favorite = false
    private(set) var isPlaying = false
    private(set) var speed = 1.0
    /// Loop region (current bar or a dragged, beat-snapped range); `nil` = off.
    private(set) var loop: Range<Double>?
    private(set) var chordsMode = ChordsMode.full
    /// Semitones, −6…+6. Display only in v1 (not audio pitch).
    private(set) var transpose = 0
    private(set) var showNoteNames = true
    private(set) var visualizer = Visualizer.piano
    /// Playhead in seconds, published on every tick and seek.
    private(set) var position = 0.0
    /// Index into `analysis.chords` of the chord under the playhead. Changes only at chord boundaries.
    private(set) var chordIndex: Int?
    /// The next real chord and the beats until it. Changes only on beats and chord boundaries.
    private(set) var upcoming: UpNext?

    @ObservationIgnored private let loadTrack: TrackLoader
    @ObservationIgnored private let voicings: ChordVoicings?
    @ObservationIgnored private let makeEngine: EngineFactory
    @ObservationIgnored private let saveFavorite: FavoriteSaver
    @ObservationIgnored private let tickInterval: Duration?
    @ObservationIgnored private var engine: (any PlaybackEngine)?
    @ObservationIgnored private var clockTask: Task<Void, Never>?
    @ObservationIgnored private var loadStarted = false

    /// - Parameter tickInterval: how often the clock ticks while playing; `nil` = only when
    ///   `tick()` is called (tests).
    init(
        loadTrack: @escaping TrackLoader,
        voicings: ChordVoicings?,
        makeEngine: @escaping EngineFactory,
        saveFavorite: @escaping FavoriteSaver = { _ in },
        tickInterval: Duration? = .milliseconds(16)
    ) {
        self.loadTrack = loadTrack
        self.voicings = voicings
        self.makeEngine = makeEngine
        self.saveFavorite = saveFavorite
        self.tickInterval = tickInterval
    }

    // MARK: - Derived

    var analysis: Analysis? { track?.analysis }

    var duration: Double { analysis?.meta.durationSec ?? 0 }

    /// Video is only for youtube sources (DESIGN.md §3.7 item 6).
    var videoAvailable: Bool { analysis?.meta.source == .youtube }

    /// Names and notes with the current Chords and Transpose settings.
    var chords: ChordDisplay {
        ChordDisplay(key: analysis?.key, transpose: transpose, basic: chordsMode == .basic, voicings: voicings)
    }

    /// The label of the chord under the playhead (`N`/`X` included).
    var currentLabel: String? {
        guard let chordIndex, let analysis, analysis.chords.indices.contains(chordIndex) else { return nil }
        return analysis.chords[chordIndex].label
    }

    /// The engine clock right now. Not observed: the timeline reads it every frame while playing.
    var livePosition: Double {
        engine?.position ?? position
    }

    // MARK: - Bindings (`@Bindable`); setting goes through the same rules as the methods.

    /// The seek slider.
    var seekPosition: Double {
        get { min(max(position, 0), duration) }
        set { seek(to: newValue) }
    }

    var speedChoice: Double {
        get { speed }
        set { setSpeed(newValue) }
    }

    var transposeChoice: Int {
        get { transpose }
        set { setTranspose(newValue) }
    }

    var visualizerChoice: Visualizer {
        get { visualizer }
        set { selectVisualizer(newValue) }
    }

    // MARK: - Loading

    /// Loads the track once and creates its engine.
    func load() async {
        guard loadStarted == false else { return }
        loadStarted = true
        let loaded = await loadTrack()
        if let loaded {
            let engine = makeEngine(loaded)
            engine.setSpeed(speed)
            self.engine = engine
        }
        track = loaded
        favorite = loaded?.favorite ?? false
        isLoading = false
        updateDerived()
    }

    // MARK: - Clock

    /// Reads the engine clock: publishes the position, wraps the loop, stops at the end.
    func tick() {
        guard let engine else { return }
        var t = engine.position
        if let loop, engine.isPlaying, t >= loop.upperBound || t < loop.lowerBound - Self.loopEntryTolerance {
            engine.seek(to: loop.lowerBound)
            t = loop.lowerBound
        } else if loop == nil, engine.isPlaying, t >= duration {
            engine.pause()
            t = duration
        }
        setPosition(t)
        if engine.isPlaying != isPlaying {
            isPlaying = engine.isPlaying
        }
    }

    private func startClock() {
        clockTask?.cancel()
        guard let tickInterval else { return }
        clockTask = Task { [weak self] in
            while Task.isCancelled == false, self?.clockTick() == true {
                try? await Task.sleep(for: tickInterval)
            }
        }
    }

    private func stopClock() {
        clockTask?.cancel()
        clockTask = nil
    }

    /// One clock tick; `false` once playback has stopped.
    private func clockTick() -> Bool {
        tick()
        return isPlaying
    }

    // MARK: - Transport

    func togglePlay() {
        guard let engine else { return }
        if engine.isPlaying {
            engine.pause()
        } else {
            if position >= duration - Self.endTolerance {
                seek(to: loop?.lowerBound ?? 0)
            }
            engine.play()
        }
        isPlaying = engine.isPlaying
        tick()
        if isPlaying {
            startClock()
        } else {
            stopClock()
        }
    }

    func pause() {
        guard let engine, engine.isPlaying else { return }
        togglePlay()
    }

    func seek(to seconds: Double) {
        guard let engine else { return }
        let t = min(max(seconds, 0), duration)
        engine.seek(to: t)
        setPosition(t)
    }

    func previousChord() {
        guard let analysis, let target = previousChordStart(at: position, in: analysis) else { return }
        seek(to: target)
    }

    func nextChord() {
        guard let analysis, let target = nextChordStart(at: position, in: analysis) else { return }
        seek(to: target)
    }

    /// Snaps to 0.05 steps within 0.5–1.25×.
    func setSpeed(_ value: Double) {
        let snapped = min(max((value * 20).rounded() / 20, Self.minSpeed), Self.maxSpeed)
        engine?.setSpeed(snapped)
        speed = snapped
    }

    /// Tap on the speed button: 1.00× ⇄ 0.75×.
    func toggleSpeed() {
        setSpeed(speed == 1 ? Self.practiceSpeed : 1)
    }

    /// Loop off → loop the current bar; loop on → clear it.
    func toggleLoop() {
        if loop != nil {
            clearLoop()
        } else if let analysis, let bar = barRange(at: position, in: analysis) {
            setLoop(bar)
        }
    }

    /// A custom range (from dragging on the chord track); the playhead jumps in if it's outside.
    func setLoop(_ range: Range<Double>) {
        loop = range
        if range.contains(position) == false {
            seek(to: range.lowerBound)
        }
    }

    func clearLoop() {
        loop = nil
    }

    // MARK: - Display settings

    func toggleChordsMode() {
        chordsMode = chordsMode == .basic ? .full : .basic
    }

    func setTranspose(_ semitones: Int) {
        transpose = min(max(semitones, -Self.maxTranspose), Self.maxTranspose)
    }

    func toggleNoteNames() {
        showNoteNames.toggle()
    }

    func selectVisualizer(_ newValue: Visualizer) {
        guard newValue == .piano || videoAvailable else { return }
        visualizer = newValue
    }

    func toggleFavorite() {
        favorite.toggle()
        saveFavorite(favorite)
    }

    // MARK: - Private

    private func setPosition(_ t: Double) {
        position = t
        updateDerived()
    }

    /// Updates the chord and up-next only when they change, so views that show them don't
    /// redraw on every tick.
    private func updateDerived() {
        guard let analysis else { return }
        let index = currentState(at: position, in: analysis).chordIndex
        if index != chordIndex {
            chordIndex = index
        }
        let next = upNext(at: position, in: analysis)
        if next != upcoming {
            upcoming = next
        }
    }
}
