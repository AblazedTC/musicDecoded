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
    /// The view switcher's view when the player opens (DESIGN.md §3.7.1).
    var defaultView = PlayerViewMode.timeline
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
    private static let defaultViewKey = "settings.defaultView"
    /// Replaced by Default view: "Default visualization" (Piano/Video) picked a panel tab that no
    /// longer exists (both panels show together in Timeline), so it has no view to carry over.
    private static let legacyVisualizerKey = "settings.visualizer"
    /// The switcher's last view, kept app-wide before Default view existed; it seeds Default view.
    private static let legacyViewModeKey = "player.viewMode"

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
        if let raw = defaults.string(forKey: Self.defaultViewKey) ?? defaults.string(forKey: Self.legacyViewModeKey),
           let view = PlayerViewMode(rawValue: raw) {
            stored.defaultView = view
        }
        if defaults.object(forKey: Self.legacyVisualizerKey) != nil || defaults.object(forKey: Self.legacyViewModeKey) != nil {
            defaults.set(stored.defaultView.rawValue, forKey: Self.defaultViewKey)
            defaults.removeObject(forKey: Self.legacyVisualizerKey)
            defaults.removeObject(forKey: Self.legacyViewModeKey)
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
        defaults.set(next.defaultView.rawValue, forKey: Self.defaultViewKey)
    }
}
