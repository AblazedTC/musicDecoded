/// A Harte-style label split into its parts: `root:quality/bass`.
/// `N` (no chord) and `X` (unknown) have no parts, so `init` returns `nil` for them.
struct ChordLabel: Sendable, Equatable {
    let root: String
    let quality: String
    let bass: String?

    init?(_ label: String) {
        guard let colon = label.firstIndex(of: ":") else { return nil }
        root = String(label[..<colon])
        let rest = label[label.index(after: colon)...]
        if let slash = rest.firstIndex(of: "/") {
            quality = String(rest[..<slash])
            bass = String(rest[rest.index(after: slash)...])
        } else {
            quality = String(rest)
            bass = nil
        }
    }
}

/// `N` and `X` are "no chord": no name, no notes, and skipped by up-next and prev/next.
func isNoChord(_ label: String) -> Bool {
    label == "N" || label == "X"
}

/// Wraps any integer into a pitch class 0...11 (handles negative transposes).
func wrapPitchClass(_ value: Int) -> Int {
    ((value % 12) + 12) % 12
}
