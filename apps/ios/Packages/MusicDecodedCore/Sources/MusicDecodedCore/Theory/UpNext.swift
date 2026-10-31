/// The next real chord (never `N`/`X`) and how many beats until it starts.
public struct UpNext: Sendable, Equatable {
    /// Index into `Analysis.chords`.
    public let chordIndex: Int
    /// Number of beats with `playhead < beat.t ≤ chord.start`.
    public let inBeats: Int
}
