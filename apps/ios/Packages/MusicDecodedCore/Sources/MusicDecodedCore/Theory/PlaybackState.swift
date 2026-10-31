/// Where the playhead is in an analysis. Indices point into `Analysis.chords` / `Analysis.beats`.
public struct PlaybackState: Sendable, Equatable {
    /// The segment with `start ≤ t < end`; `nil` before the first or at/after the last `end`.
    public let chordIndex: Int?
    /// The last beat with `t ≤ playhead`; `nil` before the first beat.
    public let beatIndex: Int?
    /// 0-based bar number, counted in downbeats; `nil` before the first downbeat (pickup).
    public let bar: Int?
}
