import Foundation
import MusicDecodedCore
import Observation

/// The Settings rows the player reads when it opens a track (DESIGN.md §3.9).
struct PlayerPreferences: Equatable, Sendable {
    /// Default speed choices (the player's long-press range is 0.50–1.25×).
    static let speeds: [Double] = [0.5, 0.75, 0.9, 1, 1.25]

    var defaultSpeed = 1.0
    var chordsMode = ChordsMode.full
    var accidentals = Accidentals.auto
    var visualizer = Visualizer.piano
}

/// UserDefaults-backed preferences (not secrets). Ports apps/android/.../settings/SettingsStore.kt.
@MainActor
@Observable
final class SettingsStore {
    private(set) var preferences: PlayerPreferences

    private let defaults: UserDefaults

    private static let speedKey = "settings.defaultSpeed"
    private static let chordsKey = "settings.chordsMode"
    private static let accidentalsKey = "settings.accidentals"
    private static let visualizerKey = "settings.visualizer"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        var stored = PlayerPreferences()
        if let speed = defaults.object(forKey: Self.speedKey) as? Double {
            stored.defaultSpeed = speed
        }
        if let raw = defaults.string(forKey: Self.chordsKey), let mode = ChordsMode(rawValue: raw) {
            stored.chordsMode = mode
        }
        if let raw = defaults.string(forKey: Self.accidentalsKey), let accidentals = Accidentals(rawValue: raw) {
            stored.accidentals = accidentals
        }
        if let raw = defaults.string(forKey: Self.visualizerKey), let visualizer = Visualizer(rawValue: raw) {
            stored.visualizer = visualizer
        }
        preferences = stored
    }

    func update(_ transform: (inout PlayerPreferences) -> Void) {
        var next = preferences
        transform(&next)
        preferences = next
        defaults.set(next.defaultSpeed, forKey: Self.speedKey)
        defaults.set(next.chordsMode.rawValue, forKey: Self.chordsKey)
        defaults.set(next.accidentals.rawValue, forKey: Self.accidentalsKey)
        defaults.set(next.visualizer.rawValue, forKey: Self.visualizerKey)
    }
}
