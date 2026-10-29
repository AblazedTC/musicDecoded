import Foundation
import MusicDecodedCore

/// Contract files packaged in the app (by reference from `contracts/`, see project.yml).
enum BundledContracts {
    /// `contracts/chord-voicings.json`, in every build. `nil` only if the bundle is broken.
    static let voicings: ChordVoicings? = {
        guard let url = Bundle.main.url(forResource: "chord-voicings", withExtension: "json") else { return nil }
        return try? ChordVoicings(contentsOf: url)
    }()
}

#if DEBUG
/// The contract fixtures, packaged in Debug builds only (Release excludes `*.analysis.json`),
/// so the player can be tried and previewed without the backend.
enum DebugFixtures {
    /// `meta.trackId` of `contracts/fixtures/sunnydays.analysis.json` (has real waveform peaks).
    static let sunnyDays = "fixture-sunnydays"

    private static let prefix = "fixture-"

    /// Loads `<name>.analysis.json` for a `fixture-<name>` id; `nil` if it isn't packaged.
    static func load(trackId: String, bundle: Bundle = .main) -> Analysis? {
        guard trackId.hasPrefix(prefix) else { return nil }
        let name = String(trackId.dropFirst(prefix.count))
        guard let url = bundle.url(forResource: "\(name).analysis", withExtension: "json") else { return nil }
        return try? Analysis(data: Data(contentsOf: url))
    }
}
#endif
