/// How black-key notes are spelled in chord names (DESIGN.md §4, Settings → Accidentals).
public enum Accidentals: String, Sendable, CaseIterable, Codable {
    /// Follow the key signature: flat keys use ♭, sharp keys use ♯. With no key, or in
    /// C major / A minor, use the common chart spelling (C♯ E♭ F♯ A♭ B♭).
    case auto
    case sharp
    case flat
}
